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
  demo.json            public manifest, with a sha256 per file
  demo.mp4             preview, for people
  demo_frames.npz      native-rate frames and times: what the policy is given
  private/             scene restore data, expert trajectory, seed attempts — never sent
```

```python
from zerowam_protocol import bundle

manifest, arrays = bundle.read("pool/rts-click_bell-000")  # hashes checked
policy.set_demonstration(bundle.public_arrays(arrays), info)
```

Video is stored at the benchmark's native frame rate with `times` beside it. Nothing here resamples
it: one bundle serves every model, and each runtime resamples for itself.

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
