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
- **Protocol 3.**
  - `PROTOCOL_VERSION` is 3 and both ends refuse 2 at `hello`: a fork, a runtime and the
    competition move together.
  - A client's `hello` says `protocol`, the `action_types` the benchmark executes and whether it
    `honors_observe_every`. `RemotePolicy` takes the two as arguments, defaulting to `("ee",)` and
    `False`, so a fork that records the observations a chunk produced passes
    `honors_observe_every=True` or is refused by an observing policy; a fork whose action types do
    not include the policy's is refused too. A refused client ends its own session: the server
    keeps its policy and serves the next one.
  - The reply to `hello` may carry `served` (`policy.SERVED_KEYS`: `family_sha256`,
    `family_version`, `knobs`, `weights_fingerprint`, `weights_sha256`), which a policy exposes as
    an attribute; `RemotePolicy.served` keeps it and a fork passes it to `result.write`, which
    refuses any other key. Its values are plain JSON: a knob that resolved to a numpy scalar or a
    path is refused at `hello` with an error reply (exit 1), naming the key, and `knobs` itself is
    a mapping, because a benchmark reads a knob by name.
  - A fork sends Q14's `info` keys with every demonstration: `embodiment`, `action_spec` (Q3),
    `cameras` (the observation cameras, `{name, role, w, h}` each), `demo_cameras` (the
    demonstration's channels, exactly the `frames_` arrays sent), `step_limit` and `instruction`,
    exactly `"Follow the demonstrated behavior."`; `demo_text` only where the video's source
    supplies a caption. `action_type`, `action_dim`, `action_dims` and `control_hz` at the top
    level are refused, and an `info` that breaks the schema is a `BundleSchemaError` (exit 2), at
    the client before anything is sent and at the server on every `prompt`. An `info` is plain JSON
    throughout, a fork's own keys included: a numpy scalar or a `Path` among them is a
    `BundleSchemaError` too, not a `WireError`.
  - The server holds a `prompt`'s arrays to the demonstration allow-list as well as its `info`, so
    a client that is not `RemotePolicy` cannot hand a policy the demonstrator's record (Q4); the
    session survives the refusal.
- **Smoke policies.**
  - A harness that serves `ZeroPolicy` sends `info.action_spec` (Q3), whose `state_channel` names
    the array it echoes to hold still, or names the channel itself with
    `--policy-arg state_channel=NAME`; `--policy-arg width=N` is gone.
  - A selftest that serves `ReplayPolicy` at a chunked cadence passes
    `--policy-arg observe_every=N`.
  - `ReplayPolicy` plays `private/expert.npz`'s `ee_actions`, the array Q4 §7 requires and the one
    in the Q3 space its `action_type` declares, where it played `actions` (the benchmark's native
    row, another width on RoboTwin) before. A harness that means to replay another array passes
    `--policy-arg key=NAME`.

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
- The contract says what a policy's `close()` means under `serve --max-sessions N`: it ends the
  session, not the policy - release the demonstration, the episode and whatever was cached for
  them, keep what it cost to build, the weights above all - because the next client is served on
  the same policy object. `policy.Policy`, `serve`'s "More than one unit" and the README say so.
  A runtime that read it as an unload reloaded its whole checkpoint for every unit
  (robotensor/zerowam-runtime#28). Wording only: no behaviour here changes.

### Fixed

- `serve.build_policy` closes a policy it refuses. The class is constructed before it is read, so
  one refused for its `action_type`, a missing method or an `observe_every` that means nothing was
  already holding whatever its `__init__` opened - weights, a CUDA context, an `expert.npz` - and
  was dropped without a `close`. The server exits with such a policy, so nothing there noticed; a
  caller that builds one policy after another in one process (`conformance.check_policy`, a fork's
  selftest over the runtimes it ships) held it for the rest of the run. A close that blows up is
  dropped, so the refusal still says why the policy could not be built. (#12)
- Five refusals no longer escape as another class, each found by the contract-v2 reviews
  (#23): `bundle.write` into a directory it cannot write (a parent that is a file, no permission, a
  full disk) raises `BundleError`, not `NotADirectoryError`, as its module promises;
  `bundle.read`, `bundle.digest`, `result.write` and `result.read` given a path the OS will not
  look up (no search permission, a name too long) raise `BundleError` or `ValueError`, where
  pathlib's `is_symlink`/`is_dir`/`is_file` probes before each wrap let the `OSError` through;
  `conventions.check_observation` lists a ragged or unconvertible `frames_*` as one more C-P3
  problem instead of raising numpy's own `ValueError` or `TypeError`; `RemotePolicy` refuses an
  infinite or out-of-range `timeout_s`, `prompt_timeout_s` or `act_timeout_s` with a `ValueError`
  naming it, where it was an `OverflowError` from the socket or a deadline thread that died; and a
  policy whose `observe_every` breaks after it was built is answered at the next `hello` with an
  error reply (exit 1), as a broken `served` is, instead of killing the server with the client
  hearing only that it went away. (#23)
- `result.write` refuses a `result.json` that is a symlink, as `result.read` and the video already
  did, instead of writing the unit's result through it and outside its run directory; and
  `result.write` and `result.read` hold `served["knobs"]` to the mapping `hello` holds it to, so a
  result cannot record a `knobs` no reply carried. (#23)
- A kept server neither leaks per session nor exits 0 when a client hangs up mid-call. Each session
  duplicated the connection's descriptor into a hang-up watch and started a thread that waited
  forever, and nothing released either: 51 sessions held 55 descriptors and 75 threads where one
  held 5 and 25, and a full evaluation of 1,490 units passes a 1024-descriptor limit. A session now
  closes its watch, so a server's descriptors and threads after many sessions are what they were
  after the first. A hang-up while a policy call runs exits `EXIT_HUNGUP` (3), its own status under
  any `--max-sessions`, so a supervisor (C20) tells a lost call from a clean finish instead of
  seeing the 0 of a server that finished its work. (#9)
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

- `zerowam_protocol.conformance`: the suite a consumer runs against itself, shipped in the wheel and
  needing numpy alone. There was none: the plan puts conformance tests here (§2.1, §12.2), the
  helpers that existed were `tests/protocol_testing.py`, which nothing installs, and so no fork,
  runtime or harness could check the contract except by rewriting it. `check_action_spec` holds a
  declared space to Q3 and P9 (C-P1), as one exception type, and returns the arms' slices;
  `hold_still(spec, position=..., gripper=...)` builds a row of the declared layout and holds it to
  C-P2 and C-P3; `check_policy` builds a `module:Class` through `serve.build_policy`, the server's
  own builder, resets it from `seed` and drives it with `calls` different observations, so every
  answer passes `serve.checked_action`, `conventions.check_chunk`, the wire encoder and, for a
  policy that declares `observe_every`, `observe.check_chunk` on a chunk sent the stack it
  produced, and with `repeat=True` the same seed is driven twice and must answer the same - each
  answer is copied as it is taken, so a policy answering from a buffer it reuses is caught in
  process as it is over the socket, and two NaNs in the same place are the same answer; a policy
  built here is closed however the check ends, what it declares (`action_type`, `observe_every`,
  `served`) being refused included, because it outlives the check where a served one goes with its
  server, and every exception the policy's own calls raise is a `ConformanceError`, as it is a
  `PolicyUnavailable` over the socket; `check_served` resets the same policy from `seed` and
  drives it, held to the `observe_every` it declared, through `python -m zerowam_protocol.serve`
  and `RemotePolicy` in two processes and requires exit 0, killing and refusing a server that
  outlives its client; both refuse the declared space (C-P1), the demonstration arrays (Q4) and
  the `info` beside them (Q14) before a policy is built or a server is started;
  `check_bundle` reads a directory as bundle v2 and adds the one thing reading it does not settle,
  that the demonstration fits in one `prompt` (counted from the arrays' shapes, so no copy of the
  bundle is made), and `check_result` reads a `result.json` as result v2 and, given the bundle
  (read as bundle v2 too), holds the two to each other (`unit_id`, the bundle's digest as
  `demo_sha256`, the task config, the fork commit). `demonstration(spec)` and
  `observation(spec, cameras)` build what the checks send, so a runtime with no benchmark beside it
  can still be driven; what a policy is handed is read-only as the server's arrays are, a
  demonstration passed in as `demo=` included, as a view that leaves the caller's own arrays
  writeable. What the module adds raises the new `ConformanceError`; what it wraps keeps raising
  `BundleSchemaError`, `BundleError` and `PolicyUnavailable`, so a fork's exit statuses do not
  change. The consumers' own adoption lands with them (RT16, RC11, RU9). (#12)
- Protocol 3: the `hello` exchange records what was served and what the client supports. The reply
  may carry `served` - the family's sha and version, the resolved knobs, the weights' fingerprint
  and sha (`policy.SERVED_KEYS`, which a policy exposes as an attribute) - so `result.json` says
  what produced it, an operator's `--knobs` included; `result.write` and `result.read` hold the
  field to those keys. The client's `hello` says which protocol it speaks, which action types the
  benchmark executes and whether it honours a policy's `observe_every`, and the server refuses a
  client that cannot drive its policy rather than answering one that would run blind: an
  `observe_every=4` policy sent an unstacked observation returned an (8, 16) chunk with no error.
  Both ends refuse protocol 2. A refused client ends only its own session, so a kept server goes on
  to the next one, and a client that connects and leaves without a `hello` no longer stops the
  server either; a refused session ends with exit 1 and counts against `--max-sessions`, which the
  exit-status table now says. (#8)
- `zerowam_protocol.info`: the keys a demonstration's `info` carries (decision Q14), as
  `REQUIRED_KEYS` (`embodiment`, `action_spec`, `cameras`, `demo_cameras`, `step_limit`,
  `instruction`), the optional `demo_text`, `INSTRUCTION` and `check_info(info, arrays)`. `info`
  was free JSON, and the forks disagreed: RoboTwin sent no `action_spec` or `control_hz` and spread
  `action_type`, `action_dim` and `action_dims` over the top level, RoboCasa spread `action_spec`
  there, and on HumanGen RoboTwin's `cameras` named the demonstration channel instead of the
  robot's observation cameras. The action space is `action_spec`'s alone now (`action_type`,
  `action_dim`, `action_dims` and `control_hz` at the top level are refused), `cameras` is the
  observation cameras as `{name, role, w, h}` and `demo_cameras` the demonstration's own channels,
  which must name exactly the `frames_` arrays sent. Channel names stay free strings: nothing here
  validates one. `RemotePolicy.set_demonstration` refuses a `BundleSchemaError` before a byte is
  sent, and the server refuses on every `prompt`, so a client that is not `RemotePolicy` cannot
  hand a policy another `info`. (#13)
- The smoke policies fit the `Policy` protocol and their own docs. `isinstance(ZeroPolicy(),
  Policy)` was False, because a `runtime_checkable` Protocol checks every member it declares and
  `observe_every` was declared although it is optional; it is documented instead, so a policy passes
  with or without it. `ZeroPolicy` now holds still by echoing the state channel
  `info.action_spec.state_channel` names (Q3) rather than sending zeros, which under `ee` command
  the frame origin with an invalid quaternion; it takes `--policy-arg state_channel=NAME` and
  refuses a demonstration or observation that leaves it nothing to echo, and its `width` argument is
  gone. `ReplayPolicy` takes `--policy-arg observe_every=N`, declares it at `hello` and answers with
  chunks of N actions, which `observe.check_chunk` accepts, so a fork's selftest covers N > 0. (#10)
  That hold is now measured rather than asserted, in both simulators and on two units each, which
  is a scale and not a bound: over RoboTwin `click_bell`'s 400 actions the end effector moved under
  half a millimetre from its first pose, over robocasa `OpenDrawer`'s 750 about a millimetre with
  the gripper within 0.014 of the opening it was shown, and every unit ended `failure`. `stubs`
  says what is left - a creep of about a micrometre per action on RoboTwin, the echo following the
  arm's own settling, and a position-only measure that leaves orientation drift out - so a harness
  knows what to expect of its smoke epoch. (#15)
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
  interval is within `UNIFORM_RTOL` of the mean plus `UNIFORM_ULPS` ulps of the times, and `null`
  when the times are uneven, a single frame or span no time. Plan §4.4 lists them and no fork wrote
  them; `read` refuses values that disagree with the times. (#5)
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
