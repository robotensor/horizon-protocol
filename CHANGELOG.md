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

### Changed

- A policy is given a demonstration's video and times only (decision Q4): the allow-list is
  `PUBLIC_PREFIXES = ("frames_",)`, `PUBLIC_NAMES = ("times",)`, and `qpos`, `endpose` and `actions`
  are no longer public. `bundle.write` and `bundle.read` refuse any other array with the new
  `BundleSchemaError`, a `BundleError` that names the array and Q4; `public_arrays` keeps allow-listed
  names only. On `same_as_demo` axes those arrays were the expert's trajectory for the very scene a
  policy is scored in. (#14)
- A bundle's digest covers demo.json and every file under `private/`. `demo.json` records
  `private_files {path: sha256}` for every file under `private/`, nested ones included;
  `bundle.digest(bundle_dir)` is the sha256 of `demo.json`'s bytes and so covers the whole unit;
  `bundle.read` refuses a private file that is changed, missing, unlisted or a symlink. An edited
  `scene_seed` or `private/scene.json` passed `read` before, although RoboTwin evaluates from them.
  `BUNDLE_VERSION` is 2, with `unit_id` and `action_spec` (Q3) required in `demo.json`. (#4)
- A bundle's arrays are compressed: an uncompressed click_bell unit was 103 MB, compressed 22 MB.

### Fixed

- `bundle.write` and `bundle.read` enforce the schema they describe, and every read failure is a
  `BundleError`. `write` accepted float64 frames of shape (5, 4, 4) with two timestamps, pickled
  object arrays into the npz, and took `camera={}`; `read` raised `FileNotFoundError` or numpy's
  `ValueError`, which the forks do not catch, and opened whatever path `files` named, `../` included.
  Now frames, `times`, object dtypes, `camera` (Q13) and `cameras` (Q4) are refused alike at write
  and read with `BundleSchemaError`; `files` names the bundle's own public files only; the npz's
  names and dtypes are read from its headers before anything is loaded. (#3)

### Added

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
