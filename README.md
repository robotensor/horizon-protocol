# zerowam-protocol

The protocol of the [Zero-WAM competition](https://github.com/robotensor): the socket between a
benchmark and a served model, and the two files they exchange around it.

A benchmark cannot import the model it evaluates — the simulator and the model pin different stacks,
and they often run on different machines — so the two meet on a socket carrying named arrays. This
distribution is both ends of that socket. It knows no benchmark, no channel name, no simulator and
no model, and depends on numpy alone.

```bash
pip install zerowam-protocol
```

## Serving a policy

```bash
export ZEROWAM_AUTHKEY=$(python -c 'import secrets; print(secrets.token_bytes(32).hex())')
python -m zerowam_protocol.serve --policy my_runtime:Policy --policy-arg checkpoint=/models/mine \
    --address 127.0.0.1:7100 --authkey-env ZEROWAM_AUTHKEY --log-file policy.log
```

A policy is a plain class with `action_type`, `reset(seed)`, `set_demonstration(arrays, info)` and
`act(observation)`; `close()` is optional. Arrays arrive read-only and go back as bool, integer or
float numpy arrays.

`close()` ends the session, not the policy: it releases what the session held - the demonstration,
the episode, whatever was cached for them - and keeps what it cost to build, the weights above all.
With `--max-sessions N` the next client is served on the same policy object, so a policy that
unloads its model here reloads it for every unit.

`--max-sessions N` serves N clients one after another, keeping the policy built for the first: a
submission is evaluated over many units and its weights are loaded once. Each session releases what
it held, so a server kept for a whole evaluation ends with the descriptors and threads it had after
the first unit. The exit status says how the last session ended:

| Status | Meaning |
|---|---|
| 0 | every session ended cleanly: `close`, a hang-up between calls, or idle too long |
| 1 | the policy could not be built, a client it cannot be driven by was refused (counting against `--max-sessions` like any other session), or a malformed message or another failure ended a session |
| 2 | serving never started: arguments, key or address |
| 3 | the client hung up **while a policy call was running**, under any `--max-sessions` |

3 is its own status because a supervisor must tell it from a clean finish: the answer is lost, the
call may never return, and the policy goes with the process, so the next unit needs a server
started again.

## Driving one

```python
from zerowam_protocol.client import RemotePolicy

with RemotePolicy(
    "127.0.0.1:7100",
    authkey,
    timeout_s=60.0,  # connect, authenticate, hello, reset, close
    prompt_timeout_s=600.0,  # set_demonstration: the first one loads the weights
    act_timeout_s=30.0,  # one act
    action_types=("ee",),  # the action types this benchmark executes
    honors_observe_every=True,  # it records the observations a chunk produced
    log_file="policy.log",
) as policy:
    policy.hello()  # protocol, action_type, observe_every, policy, served
    policy.set_demonstration(demo_arrays, info)  # one demonstration, named arrays
    policy.reset(seed)
    action = policy.act(observation)["action"]  # (A,) or a chunk (H, A)
```

A policy that must see what its own chunk did declares `observe_every = N`. The benchmark then
records an observation after every N-th action and sends them, stacked, with the next `act`
(`zerowam_protocol.observe.stack`); the first `act` of an episode carries the initial observation
alone. A chunk whose length is not a multiple of N is refused (`observe.check_chunk`).

`PROTOCOL_VERSION` is 3, and both ends refuse anything else at `hello`. The client's greeting says
which action types this benchmark executes and whether it records those observations; a server
whose policy the client cannot drive — another action type, or a cadence the benchmark ignores —
refuses that client with an error and serves the next one, rather than answering a policy that
would run blind. Both declarations default to the cautious answer (`("ee",)`, `False`), so a
benchmark says what it does deliberately.

The reply may carry `served`, what the server says this process serves —
`zerowam_protocol.policy.SERVED_KEYS`: `family_sha256`, `family_version`, `knobs`,
`weights_fingerprint`, `weights_sha256` — which a policy exposes as an attribute of its own. It is
kept as `policy.served`, and the benchmark records it in `result.json` unchanged, so every result
says what produced it, an operator's resolved knobs included. Its values are plain JSON: a knob
that resolved to a numpy scalar is refused at `hello` with an error reply (exit 1), not left to
the encoder, and `knobs` itself must be a mapping, because a benchmark reads a knob by name.

The three timeouts are separate because the three calls cost different things: a runtime loads its
weights inside the first `set_demonstration`, so an act budget of 30 s would fail it, and giving
every call the prompt's budget instead lets a policy sit in one `act` for ten minutes. Both
per-call budgets default to `timeout_s`, and each call is bounded by its own alone.

Every way the policy can fail — an error reply, a timeout, a hang-up, a malformed message — raises
`PolicyUnavailable`, and the benchmark decides what it costs the unit.

The `info` beside a demonstration carries the keys decision Q14 names, and both ends check them
(`zerowam_protocol.info.check_info`):

| Key | What it says |
|---|---|
| `embodiment` | the robot, which a runtime looks up in its own table |
| `action_spec` | the action and observation space (decision Q3) |
| `cameras` | the evaluated robot's **observation** cameras, `{name, role, w, h}` each |
| `demo_cameras` | the **demonstration**'s own channels, exactly the `frames_` arrays sent |
| `step_limit` | how many actions the unit allows |
| `instruction` | always `"Follow the demonstrated behavior."`, never task language |
| `demo_text` | optional: the video's own caption, where its source supplies one |

The action space is `action_spec`'s alone, so `action_type`, `action_dim`, `action_dims` and
`control_hz` at the top level are refused; channel names inside stay free strings, which nothing
here validates. An `info` that breaks the schema is a `BundleSchemaError` before a byte is sent —
the benchmark's own error (exit 2), never `PolicyUnavailable` — and the server checks it too, so a
client that is not `RemotePolicy` cannot hand a policy another `info`.

## Conventions and demonstrations

**The convention (decision Q3).** Every number a benchmark and a model exchange means the same thing
on every benchmark, and each benchmark fork converts its simulator's values to it, in both
directions. An `ee` action, and the observation's state channel, is 8 numbers per arm,
`[x, y, z, qw, qx, qy, qz, g]`:

- the position in **metres**, in the frame `info.action_spec.frame` names (`world`, or
  `robot_base`, the base link of the arm's chain);
- the orientation as a unit quaternion, **scalar first `(qw, qx, qy, qz)`**, rotating that frame
  into the tool frame; q and −q are the same rotation;
- the gripper `g` in **[0, 1], 0 = fully closed, 1 = fully open**, a position target, so echoing
  the observed state holds the robot still.

Two arms go left then right (A = 16). A one-armed robot is a single block (A = 8), declared
`arms: ["right"]`; the name is a label on the wire and never picks a model slot, which each model
runtime's own table does (Zero-WAM puts a single arm in its slot 0). The benchmark declares what is
native to its robot (frame, tool, execution, gripper semantics) in `info.action_spec`;
`zerowam_protocol.conventions` holds the constants and the checks (`check_action_spec`,
`check_chunk`, `check_observation`, `arm_slices`, `same_rotation`).

Two pages are the contract every benchmark and every model family implements; read them before
adding either:

- [`docs/conventions.md`](docs/conventions.md) — what every action and observation number means
  (metres, quaternion `wxyz`, gripper 0 closed … 1 open, 8 per arm), what a benchmark declares in
  `info.action_spec`, and what a model runtime maps for itself (decision Q3).
- [`docs/demonstrations.md`](docs/demonstrations.md) — what a policy receives from a demonstration
  (video frames, `times`, and a HumanGen video's caption) and what stays under `private/` (the
  demonstrator's state and actions) (decision Q4).

Both decisions are recorded in `robotensor/zerowam-competition` `docs/decisions.md`.

## The demonstration bundle

One directory per unit, written once per epoch and handed unchanged to every submission:

```
<unit_id>/
  demo.json            manifest, with a sha256 of every other file; never sent to a policy
  demo.mp4             preview, for people
  demo_frames.npz      native-rate frames and times, nothing else: what the policy is given
  private/             scene restore data, the demonstrator's state and actions — never sent
```

```python
from zerowam_protocol import bundle

# A benchmark writes one; whatever else goes under private/ (a scene) is there before this call.
bundle.write(unit_dir, manifest=manifest, arrays=demo_arrays, private=scene, expert=record)

manifest, arrays = bundle.read("pool/rts-click_bell-000")  # every file checked
policy.set_demonstration(bundle.public_arrays(arrays), info)
demo_sha256 = bundle.digest("pool/rts-click_bell-000")  # the whole bundle's digest
```

`demo.json` hashes every other file, `private/` included, so its own sha256,
`bundle.digest(bundle_dir)`, pins the whole unit, the scene it is evaluated in included.
`bundle.read` holds the directory to `demo.json` exactly: a file missing, changed, added after
`write` or symlinked is a `BundleError` - `demo.json` itself included, which no hash of its own
would catch. A manifest carries the unit's `unit_id` and the
`action_spec` its private pose arrays follow (decision Q3, checked with
`conventions.check_action_spec`); `BUNDLE_VERSION` is 2, and `read` refuses any other.

A policy is given a demonstration's video, `frames_<camera>`, and its `times`, and nothing else
(decision Q4): the demonstrator's state and actions (`qpos`, `endpose`, `actions`, `ee_actions`,
`states`) stay in `private/expert.npz`. The allow-list, `bundle.PUBLIC_PREFIXES` and
`bundle.PUBLIC_NAMES`, is closed: `bundle.write` and `bundle.read` refuse any other array with
`BundleSchemaError`, a `BundleError` that a benchmark maps to exit 2 (fix the writer), where any
other `BundleError` is exit 4 (rebuild the unit). `RemotePolicy.set_demonstration` refuses them too,
before a byte is sent, and the server refuses them again on every `prompt`, so the demonstrator's
record reaches a policy from no client, `RemotePolicy` or not.

`write` and `read` hold a bundle to one schema: every `frames_<camera>` uint8 RGB `(T, H, W, 3)`
with one T ≥ 2, `times` float64 `(T,)` and never decreasing, `camera` with exactly the fields
decision Q13 sets for its `demo_source` and with `w` and `h` the primary camera's own frames',
and `cameras` naming the `frames_` arrays with the primary camera first. Nothing is ever pickled: an object array is refused. Everything `read` cannot trust -
a hash that does not match, a missing file, a manifest or an npz that cannot be read, a number
JSON carries that no double can hold - is a `BundleError`, never an `OSError`, an `OverflowError`
or numpy's own `ValueError`.

A model family declares what it reads from a demonstration in its family file's `inputs` block,
from one vocabulary that binds every family: `inputs.demonstration` holds `video` and may hold
`caption` (`bundle.DEMONSTRATION_INPUTS`), and `inputs.prompt_language` is `none`, `generic` or
`demonstration_caption` (`bundle.PROMPT_LANGUAGES`). `bundle.check_demonstration_inputs(inputs)`
lists every problem, citing Q4; the competition's config check and each runtime's loader call it.

Video is stored at the benchmark's native frame rate with `times` beside it. Nothing here resamples
it: one bundle serves every model, and each runtime resamples for itself. `write` records what the
times give in `demo.json`: `n_frames`, `duration_s` and `fps`, which is `null` when the times are
not uniformly spaced (`bundle.frame_timing`).

## Results

Every evaluated unit writes a `result.json`, however it ended:

```python
from zerowam_protocol import bundle, result

# rollout.mp4, if any, is written into out_dir first: write hashes it.
result.write(
    out_dir,
    unit_id=manifest["unit_id"],
    demo_sha256=bundle.digest(bundle_dir),  # the whole bundle's digest
    outcome="success",
    task_config=manifest["task_config"],
    task_config_sha256=manifest["task_config_sha256"],
    fork_commit=fork_commit,
    timing={"setup_s": 4.1, "policy_s": 61.0, "sim_s": 118.2, "total_s": 184.0},
    steps=214,
    step_limit=400,
    fingerprint_ok=True,
    policy_calls=7,
)
record = result.read(out_dir)  # the same rules again
```

`outcome` is `success`, `failure` (the model's own side, including its errors) or `void` with a
`void_cause` of `harness` or `runtime` (decision Q6); `write` and `read` both refuse a void without
a cause and a success or failure with one. A harness void is dropped for every submission, so all
of them stay compared on the same units.

`RESULT_VERSION` is 2, and `read` refuses any other. Every field is in every result, `null` where
it does not apply, and `extra` may add keys of a fork's own but never one of them, so nothing can
overwrite an outcome. `demo_sha256` is `bundle.digest(bundle_dir)`; `task_config`,
`task_config_sha256` and `fork_commit` record what produced the result; `timing` holds exactly
`setup_s`, `policy_s`, `sim_s` and `total_s` (0.0 for a phase the unit never reached);
`rollout_mp4_sha256` is filled by `write` from the `rollout.mp4` beside the result, and `read`
checks one it finds there. Neither `result.json` nor `rollout.mp4` is read through a symlink: a run
directory holds its own result and its own video. A unit run against a stub records `stub_policy` (`zero` or `replay`),
which the competition refuses to score outside a dry run (decision Q4); `served` records what the
server said it served.

## Smoke policies

`zerowam_protocol.stubs` holds two policies that need no model: `ZeroPolicy`, which holds still, and
`ReplayPolicy`, which plays a bundle's `private/expert.npz` `ee_actions` — the demonstration in
the Q3 convention — and must succeed. They are how a benchmark's whole
chain — demonstration, hashing, scene restore, socket, result — is tested in minutes on a laptop
GPU.

An action is an absolute target, so holding still is not an action of zeros: `ZeroPolicy` echoes the
observed state channel, the array `info.action_spec.state_channel` names, and refuses a
demonstration or an observation that leaves it nothing to echo (`--policy-arg state_channel=NAME`
names it where the info does not). `ReplayPolicy` takes `--policy-arg observe_every=N`, declares it
at `hello` and answers with chunks of N actions, so a selftest covers the chunked cadence too; it
plays `ee_actions` unless `--policy-arg key=NAME` names another array of the same npz.

Both are `Policy`: the protocol declares `action_type`, `reset`, `set_demonstration` and `act`, and
nothing else, so `observe_every` and `close` stay optional for `isinstance`.

## Development

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

Ported from `icil-policy` in
[ICIL-competition-orchestrator](https://github.com/robotensor/ICIL-competition-orchestrator),
Apache-2.0. Licensed under [Apache-2.0](LICENSE).
