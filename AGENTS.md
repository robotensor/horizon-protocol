# zerowam-protocol — the socket and the files of the Zero-WAM competition

Python 3.10, package `zerowam_protocol` under `src/`. It is both ends of the socket between a
benchmark and a served model, plus the two file formats they exchange: the demonstration bundle and
the unit result. It holds **no benchmark, no model and no competition logic**.

Consumers: `robotensor/RoboTwin` and `robotensor/robocasa` (the `robotensor_bench` packages),
`robotensor/zerowam-runtime` (serves the model), `robotensor/zerowam-competition` (builds the pool,
reads the results). A change here is a change to all of them.

## The pieces

- `policy.py` — `Policy`, what a runtime implements, and `ACTION_TYPES`.
- `serve.py` — `python -m zerowam_protocol.serve --policy MODULE:CLASS`, one policy, one client.
- `client.py` — `RemotePolicy`, what a benchmark drives it with. Every failure is `PolicyUnavailable`.
- `wire.py` — the message format: a JSON header frame, then one raw little-endian frame per array.
- `observe.py` — what a policy that declares `observe_every` is sent between chunks: `stack`,
  `check_chunk`.
- `conventions.py` — decision Q3 as code: the constants every number follows, `check_action_spec`,
  `check_chunk(a, spec)`, `check_observation`, `arm_slices`, `same_rotation`.
- `bundle.py` — the demonstration bundle: `demo.json`, `demo_frames.npz`, `private/`.
- `result.py` — `result.json`: `outcome` (`success`/`failure`/`void`) and `void_cause`.
- `stubs.py` — `ZeroPolicy` and `ReplayPolicy`, the model-free smoke tests.

The wire, client and serve modules are ported from `robotensor/ICIL-competition-orchestrator`
(`packages/icil-policy`, Apache-2.0); read the original before reworking one of them. What changed:
a policy is named by `module:Class` instead of a repository manifest, because a submission here is
weights only and the runtime is ours.

## Commands

- `uv venv --python 3.10 .venv && uv pip install -e ".[dev]"`
- `ruff check . && ruff format --check .`
- `pytest -q`

## Rules

- **The conventions are `docs/conventions.md` (decision Q3) and what a policy receives is
  `docs/demonstrations.md` (decision Q4).** Code here, in the forks and in every runtime follows them;
  a change to either is a new decision in `robotensor/zerowam-competition` `docs/decisions.md`, then
  a contract change here.
- The wire carries named arrays and JSON fields only. Never pickle; object dtypes are refused at
  both ends. Both halves of the socket run beside foreign pins, so the only dependency is numpy.
- Nothing here knows a channel name. `frames_head`, `qpos`, `actions` are the benchmark's words;
  this package neither defines nor validates them. What it does own (Q3, Q4): the convention every
  number follows, the `action_spec` vocabulary a benchmark declares its state channel in, the public
  allow-list of a bundle (`frames_*`, `times`) and the demonstration-input vocabulary a family
  declares from.
- No privileged data crosses the socket. A bundle's `private/` is never read by the code that talks
  to a policy; `bundle.public_arrays` is the only thing given to one.
- A demonstration is stored at the benchmark's native frame rate, with `times` beside it. Never
  resample here: a bundle serves every model, and each runtime resamples for itself.
- One bundle is one unit and is immutable once written. `demo.json` hashes its files; `read`
  verifies them by default.
- Every evaluated unit writes a `result.json`, however it ended. A missing file is a harness
  failure, never a score.
- `ReplayPolicy` is a test instrument and is given a privileged file by the harness that already
  holds it. It must never be served to score a submission.
- A protocol or file-format change bumps `PROTOCOL_VERSION`, `BUNDLE_VERSION` or `RESULT_VERSION`
  and is a CHANGELOG entry; both sides check the version they read.

## Conventions

- Commit at completed, validated deliverables. Default to one commit per task or issue, keeping
  related implementation, tests, documentation and mechanical changes together. During ongoing
  work, batch edits and fixes in the working tree; do not commit after each file, plan step or
  test run. Each commit must pass the relevant checks and leave the repository in a working state.
- Commit title: `(feat): …`, `(fix): …`, `(refactor): …`, `(docs): …`, `(test): …`, `(chore): …`;
  imperative, lower-case after the prefix, under 72 characters, no trailing period. Body: why the
  change, not what the diff shows; short bullets; `Refs #N` for the issue it advances,
  `Closes #N` only on the commit that finishes it.
- Issues stand on their own: the title is the outcome in plain words, and the body is `## Why`,
  `## Scope` (with `Out of scope:`), `## Acceptance criteria` as a `- [ ]` checklist, and optional
  `## Notes`. A bug has `## What happened`, `## Expected` and `## Cause`. On closing, add
  `## Outcome`.
- Small changes go straight to `main`: a typo, a doc line, a version bump, a one-line fix with its
  test. Anything that changes a published contract takes a branch (`short-slug`) and a PR that says
  `Closes #N`, with tests and a CHANGELOG entry.
- Rebase, do not merge `main` into a branch; delete a branch once it is merged.
