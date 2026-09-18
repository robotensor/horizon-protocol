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

## Driving one

```python
from zerowam_protocol.client import RemotePolicy

with RemotePolicy("127.0.0.1:7100", authkey, timeout_s=60.0, log_file="policy.log") as policy:
    policy.hello()  # protocol, action_type, observe_every, policy
    policy.set_demonstration(demo_arrays, info)  # one demonstration, named arrays
    policy.reset(seed)
    action = policy.act(observation)["action"]  # (A,) or a chunk (H, A)
```

A policy that must see what its own chunk did declares `observe_every = N`. The benchmark then
records an observation after every N-th action and sends them, stacked, with the next `act`
(`zerowam_protocol.observe.stack`); the first `act` of an episode carries the initial observation
alone. A chunk whose length is not a multiple of N is refused (`observe.check_chunk`).

Every way the policy can fail — an error reply, a timeout, a hang-up, a malformed message — raises
`PolicyUnavailable`, and the benchmark decides what it costs the unit.

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
other `BundleError` is exit 4 (rebuild the unit).

`write` and `read` hold a bundle to one schema: every `frames_<camera>` uint8 RGB `(T, H, W, 3)`
with one T ≥ 2, `times` float64 `(T,)` and never decreasing, `camera` with exactly the fields
decision Q13 sets for its `demo_source` and with `w` and `h` the primary camera's own frames',
and `cameras` naming the `frames_` arrays with the primary camera first. Nothing is ever pickled: an object array is refused. Everything `read` cannot trust -
a hash that does not match, a missing file, a manifest or an npz that cannot be read, a number
JSON carries that no double can hold - is a `BundleError`, never an `OSError`, an `OverflowError`
or numpy's own `ValueError`.

Video is stored at the benchmark's native frame rate with `times` beside it. Nothing here resamples
it: one bundle serves every model, and each runtime resamples for itself. `write` records what the
times give in `demo.json`: `n_frames`, `duration_s` and `fps`, which is `null` when the times are
not uniformly spaced (`bundle.frame_timing`).

## Results

Every evaluated unit writes a `result.json`, however it ended:

```python
from zerowam_protocol import result

result.write(out_dir, unit_id=unit_id, demo_sha256=sha, outcome="success", steps=214)
```

`outcome` is `success`, `failure` (the model's own side, including its errors) or `void` with a
`void_cause` of `harness` or `runtime`. A harness void is dropped for every submission, so all of
them stay compared on the same units.

## Smoke policies

`zerowam_protocol.stubs` holds two policies that need no model: `ZeroPolicy`, which holds still, and
`ReplayPolicy`, which plays an expert trajectory and must succeed. They are how a benchmark's whole
chain — demonstration, hashing, scene restore, socket, result — is tested in minutes on a laptop
GPU.

## Development

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

Ported from `icil-policy` in
[ICIL-competition-orchestrator](https://github.com/robotensor/ICIL-competition-orchestrator),
Apache-2.0. Licensed under [Apache-2.0](LICENSE).
