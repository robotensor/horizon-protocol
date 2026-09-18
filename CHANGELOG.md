# Changelog

All notable changes to this distribution. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking: bundle v2, result v2, protocol 3

One contract release, so that every consumer (both forks, zerowam-runtime, zerowam-competition)
adopts one version set once. What a consumer must change:

- **Bundle v2.**
  - A fork writes only `frames_<camera>` and `times` in the public arrays; the demonstrator's state
    and actions (`qpos`, `endpose`, `actions`, `ee_actions`, `states`) go under `private/` (Q4).
    `bundle.write` and `bundle.read` refuse any other public array.
  - A fork catches `BundleSchemaError` before `BundleError`: exit 2 for it (fix the writer), exit 4
    for any other `BundleError` (rebuild the unit).
  - A reader of `public_arrays` gets `frames_*` and `times` only; nothing may expect `qpos`,
    `endpose` or `actions` in a demonstration.
  - `RemotePolicy.set_demonstration` refuses the same arrays `bundle.write` does, before it sends
    anything: a fork that assembles its own demonstration arrays rather than passing
    `bundle.public_arrays` gets `BundleSchemaError` (exit 2), not a served policy.
  - Frames are uint8 RGB `(T, H, W, 3)` with one T ≥ 2 and `times` float64 `(T,)`, never
    decreasing; no array is an object array.
  - `demo.json` carries `cameras` (the primary camera first, the rest in name order, exactly the
    `frames_` arrays), and `camera` carries exactly Q13's fields for its `demo_source`:
    `{name, w, h, fovy, pose}` with `pose = [x, y, z, qw, qx, qy, qz]` for `expert` and `mimicgen`,
    `{name, w, h, source, video}` for `humangen` (whose `camera.name` is `human`, the channel).
    `camera.w` and `camera.h` are the primary camera's own frames' width and height, as
    `info.cameras`' are an observation's (Q3): a lens that is not the frames' size is refused.
  - `bundle.read` raises only `BundleError`s; `bundle.digest` of an unreadable file too. No file of
    a bundle may be a symlink, `demo.json` included (the file the digest is taken of, which no hash
    of the bundle covers). A number `demo.json` carries that no double can hold - a JSON integer
    is unbounded - is a listed schema problem, never an `OverflowError`.
  - `BUNDLE_VERSION` is 2 and `bundle.read` refuses any other with `BundleSchemaError`: every pool
    built before is rebuilt.
  - `demo.json` requires `unit_id` (a reader no longer falls back to the directory name) and
    `action_spec`, which replaces `action_dims` and must pass `conventions.check_action_spec` (Q3).
  - A fork passes the demonstrator's record as `bundle.write(..., expert=arrays)` and writes
    anything else under `private/` before calling `bundle.write`; a file added under `private/`
    afterwards, or beside the bundle, fails `bundle.read`. `bundle.write` refuses a manifest that
    passes `bundle_version`, `files` or `private_files`, and a directory holding anything that is
    not part of a bundle.
  - A result's `demo_sha256` is `bundle.digest(bundle_dir)`, the sha256 of `demo.json`, not
    `files["demo_frames.npz"]`.
  - `bundle_version` is held to its type as well as its value: a manifest carrying `2.0` is
    refused, like `"2"`.
  - `demo.json` carries `n_frames`, `duration_s` and `fps`, filled in by `bundle.write` from
    `times`; a manifest that passes one is refused. `fps` is recorded when the intervals are
    uniform within `bundle.UNIFORM_RTOL` of their mean plus `bundle.UNIFORM_ULPS` ulps of the
    times, so a demonstration timed from a wall clock keeps a rate (with that grid's error in it:
    record times relative to the episode where you can).
- **Result v2.**
  - `RESULT_VERSION` is 2 and `result.read` refuses any other: a result written by an older fork is
    rewritten by rerunning the unit.
  - `result.write` takes `task_config`, `task_config_sha256`, `fork_commit` and `timing` as
    required arguments. `timing` holds exactly `setup_s`, `policy_s`, `sim_s` and `total_s`, each a
    finite number of seconds ≥ 0, on every path (0.0 for a phase the unit never reached).
  - `demo_sha256` is `bundle.digest(bundle_dir)` and `unit_id` is the manifest's, never the
    directory name; both are checked (a sha256, a non-empty string).
  - A fork writes `rollout.mp4` into the output directory before calling `result.write`, which
    records its sha256 as `rollout_mp4_sha256`; it is not an argument.
  - `extra` may not name any field of the schema (`result.REQUIRED_KEYS`).
  - A fork that serves a stub passes `stub_policy="zero"` or `"replay"` (Q4); `served` takes what
    the server's reply to `hello` says it served.
  - Every refusal from `zerowam_protocol.result` is a `ValueError`, an `out_dir` that cannot be
    written included: a fork that maps the class to its harness exit no longer sees an `OSError`.
  - A reader of `result.json` goes through `result.read`: every field is present (`null` where it
    does not apply), and a void without a cause, a success or failure with one, a timing key
    missing or extra, or a changed `rollout.mp4` is refused. Neither `result.json` nor
    `rollout.mp4` is read through a symlink: a run directory holds its own result and its own
    video.

### Changed

- A policy is given a demonstration's video and times only (decision Q4): the allow-list is
  `PUBLIC_PREFIXES = ("frames_",)`, `PUBLIC_NAMES = ("times",)`, and `qpos`, `endpose` and `actions`
  are no longer public. `bundle.write` and `bundle.read` refuse any other array with the new
  `BundleSchemaError`, a `BundleError` that names the array and Q4; `public_arrays` keeps allow-listed
  names only. On `same_as_demo` axes those arrays were the expert's trajectory for the very scene a
  policy is scored in. (#14)
- The one socket call that carries a demonstration checks the allow-list too:
  `RemotePolicy.set_demonstration` runs `bundle.check_public_arrays` and raises `BundleSchemaError`
  naming the array and Q4 before a byte reaches the policy, so a fork that builds its own arrays
  cannot leak the demonstrator's record past `bundle.read`. The refusal does not touch the
  connection and is never `PolicyUnavailable`: it is the benchmark's error (exit 2), never a
  harness void or a model failure. (#16)
- A bundle's digest covers demo.json and every file under `private/`. `demo.json` records
  `private_files {path: sha256}` for every file under `private/`, nested ones included;
  `bundle.digest(bundle_dir)` is the sha256 of `demo.json`'s bytes and so covers the whole unit;
  `bundle.read` refuses a private file that is changed, missing, unlisted or a symlink. An edited
  `scene_seed` or `private/scene.json` passed `read` before, although RoboTwin evaluates from them.
  `BUNDLE_VERSION` is 2, with `unit_id` and `action_spec` (Q3) required in `demo.json`. (#4)
- A bundle's arrays are compressed: an uncompressed click_bell unit was 103 MB, compressed 22 MB.

### Fixed

- A result's outcome can no longer be overwritten, and results record their provenance (result
  v2). `extra` was merged last, so `write(outcome="failure", extra={"outcome": "success"})` wrote a
  success; `read` accepted a void with no cause, a void caused by "model" and a success with
  `void_cause`. Now `extra` may not name a schema field, and `read` applies every rule `write`
  does (outcome and `void_cause` per Q6, types, timing). The timing keys are named, `task_config`,
  `task_config_sha256` and `fork_commit` are recorded, `write` hashes `rollout.mp4` into
  `rollout_mp4_sha256`, `demo_sha256` is the whole-bundle digest, `unit_id` is required and
  non-empty, and the optional `stub_policy` (Q4) and `served` fields exist. (#6)
- `bundle.write` and `bundle.read` enforce the schema they describe, and every read failure is a
  `BundleError`. `write` accepted float64 frames of shape (5, 4, 4) with two timestamps, pickled
  object arrays into the npz, and took `camera={}`; `read` raised `FileNotFoundError` or numpy's
  `ValueError`, which the forks do not catch, and opened whatever path `files` named, `../` included.
  Now frames, `times`, object dtypes, `camera` (Q13) and `cameras` (Q4) are refused alike at write
  and read with `BundleSchemaError`; `files` names the bundle's own public files only; the npz's
  names and dtypes are read from its headers before anything is loaded. (#3)

### Added

- `RemotePolicy` takes a timeout per kind of call: `timeout_s` still covers connecting,
  authenticating, `hello`, `reset` and `close`, `prompt_timeout_s` covers `set_demonstration` and
  `act_timeout_s` covers one `act`; both default to `timeout_s`, and each call is bounded by its own
  alone. One budget for everything meant the competition's 30 s act limit also failed the first
  prompt, in which a runtime loads its weights, so the scripts passed 600 s for every call instead.
  A fork that takes the two limits passes them here. (#7)
- A model family's demonstration inputs are checked against one vocabulary, beside the allow-list:
  `bundle.DEMONSTRATION_INPUTS = ("video", "caption")`, `bundle.PROMPT_LANGUAGES = ("none",
  "generic", "demonstration_caption")` and `bundle.check_demonstration_inputs(inputs) -> list[str]`,
  which refuses a missing `video`, `proprio`, `actions`, `task` and `demonstration_caption` without
  `caption`, each citing Q4. The check lived in zerowam-runtime, which binds only Zero-WAM; the
  competition's config check and every family loader call this one. (#17)
- `bundle.write` records `n_frames`, `duration_s` and `fps` in `demo.json`, derived from `times`
  alone by `bundle.frame_timing`: `fps` is `(T - 1) / duration_s`, rounded to 6 decimals, when every
  interval is within `UNIFORM_RTOL` of the mean, and `null` when the times are uneven, a single
  frame or span no time. Plan §4.4 lists them and no fork wrote them; `read` refuses values that
  disagree with the times. (#5)
- `zerowam_protocol.conventions`: decision Q3's convention as code. Constants for what every number
  means (`PER_ARM_LAYOUT` `[x, y, z, qw, qx, qy, qz, gripper]`, metres, scalar-first quaternions,
  gripper 0 closed … 1 open, left then right, a single arm declared `right`) and the vocabulary a
  benchmark declares its space in (`FRAMES`, `TOOL_APPROACH_AXES`, `EXECUTIONS`, `CAMERA_ROLES`,
  …); `check_action_spec` (C-P1), `check_chunk(action, spec)` (C-P2),
  `check_observation(observation, spec, cameras)` (C-P3), `arm_slices` (wire positions, never model
  slots) and `same_rotation` (up to sign). `policy.py` and the README state the convention and that
  each fork converts to it. (#11)
- Protocol 2: a policy may declare `observe_every = N`, returned in the reply to `hello`. The
  benchmark then records an observation after every N-th action of a chunk and sends them, stacked
  oldest first, with the next `act`; a chunk whose length is not a multiple of N is refused.
  `zerowam_protocol.observe` holds the rule, `stack` and `check_chunk`. Under protocol 1 a model
  that conditions on what its own chunk did - Zero-WAM does - could only run blind. N = 0, the
  default, is protocol 1's behaviour. Both ends check the version. (#1)

- The wire format, `RemotePolicy` and `python -m zerowam_protocol.serve`, ported from `icil-policy`
  in ICIL-competition-orchestrator. A policy is named by `module:Class` and built with
  `--policy-arg` values; submissions here are weights only, so there is no repository manifest.
- `bundle`: the demonstration bundle written once per unit (`demo.json`, `demo_frames.npz`,
  `private/`), with a sha256 per file and `public_arrays` as the only thing a policy is given.
- `result`: `result.json` with `outcome` and `void_cause`.
- `stubs`: `ZeroPolicy` and `ReplayPolicy`, so a benchmark's chain can be smoke-tested with no model.
- `serve --max-sessions N`: one server takes N clients one after another and keeps the policy it
  built. A submission is evaluated over many units, and loading tens of gigabytes of weights per
  unit would cost more than the units do.
