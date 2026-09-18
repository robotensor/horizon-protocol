# Demonstrations: what a policy receives, and what stays private

Normative source: robotensor/zerowam-competition docs/decisions/q4-demonstration.md (decision Q4).

This page is for someone who adds a benchmark (a fork that writes bundles) or a model family (a runtime that consumes them).
- If this page and the decision disagree, the decision wins.
- MUST, MUST NOT, SHOULD and MAY are normative.
- The action and observation conventions (layouts, frames, quaternion order, gripper direction, slots) are decision Q3's, specified in `docs/conventions.md`. §1.1 restates the few this page relies on, for reading only; where they differ, Q3 wins.

**Status.** This page describes the contract as of bundle version 2. The items that build it:

| Item | Repository | What it adds | Status |
|---|---|---|---|
| P12 | this repo | the allow-list | landed: `bundle.write`, `bundle.read` and `public_arrays` |
| P1 | this repo | `BundleSchemaError` at write and read | landed: the allow-list, object dtypes, the frames and `times` rules, `camera` (Q13) and `cameras`; every read failure is a `BundleError` |
| P2 | this repo | bundle version 2: every file hashed, `private/` included, and `demo.json`'s sha256 as the bundle's digest | landed: `bundle.write(..., expert=...)`, `private_files`, `bundle.digest(bundle_dir)` |
| P13 | this repo | the check at send | pending |
| P14 | this repo | the input vocabulary | pending |
| P10 | this repo | conformance | pending |
| P11 | this repo | `info` keys | pending |
| P4 | this repo | the optional `stub_policy` field in `result.json` | pending |
| RT3, RT9, RT11, RT16 | RoboTwin fork | exit 2, stub marking, `info` keys, tests | pending |
| RC1, RC11 | robocasa fork | exit 2, tests | pending |
| RU9, RU15 | zerowam-runtime | runtime and recipes | pending |
| C3, C10, C12 | zerowam-competition | competition | pending |

Until P13 lands, `RemotePolicy.set_demonstration` does not check the allow-list: a fork MUST send `bundle.public_arrays` of what `bundle.read` returned, and nothing else.

---

## 1. The rule

A policy receives a demonstration's **video** (`frames_<camera>`), its **times**, and three `info` keys:
- `demo_cameras`;
- `instruction`, which is always the generic sentence;
- `demo_text`, only when the video's own source supplies a caption of that video.

The **demonstrator record** is never public and reaches a policy through no message.
- **What it is.** Any numeric per-step state, command, contact or success signal of whoever performed the demonstration, and anything computed from those. The rendered RGB frames of the listed demonstration cameras, and their times, are not part of it.
- **Where it goes.** A benchmark that has such a record writes it under `private/`.

For Zero-WAM this means:
- **The runtime receives only the demonstration video, plus its caption where one exists.**
- **Simulation bundles keep the demonstrator's state and actions, privately.**

### 1.1 Conventions this page relies on (Q3; restated for reading only)

| Topic | Rule |
|---|---|
| Pose arrays (`endpose`, `ee_actions`) | 8 numbers per arm: `[x, y, z, qw, qx, qy, qz, g]`. Two arms go left then right (A = 16). A one-armed robot has a single block, declared `arms: ["right"]` (A = 8). |
| Position | Metres, in the frame named by `action_spec.frame`: `world` on RoboTwin, `robot_base` on RoboCasa. `demo.json` records the bundle's `action_spec`. Forks never convert frames. |
| Quaternion | Scalar first, `(qw, qx, qy, qz)`, a unit quaternion that rotates the declared frame into the tool frame. q and −q are both valid. |
| Gripper `g` | In [0, 1]: **0 = fully closed, 1 = fully open**. The same direction applies in observations, actions and private pose arrays. |
| Slots | A bundle, `info` and the wire never assign a model slot. An arm's name on the wire (`right`) never picks one. Slots are each model runtime's own table (for example, Zero-WAM puts a single arm in its slot 0). |
| `qpos`, `actions`, `states` | Native to the benchmark and outside the convention. No model family may consume `qpos` until a decision defines it. |
| Observation each step | `frames_<name>` for each `name` in `info.cameras` (a list of `{name, role, w, h}`), the array named by `info.action_spec.state_channel` (`endpose`), and the native `qpos`. |

---

## 2. What a bundle holds

```
<unit_id>/
  demo.json            the manifest (public only after close, Q5). Never sent to a policy; MUST NOT leave the organiser's nodes before close
  demo.mp4             preview of the demonstration, for people. MAY leave before close
  demo_frames.npz      the public arrays: frames_<camera> and times. Nothing else. Sent to the policy
  private/             never sent to a policy; MUST NOT leave the organiser's nodes before close
    scene.json         benchmark, fork commit, task, seed, resolved config, fingerprint, attempts
    scene/             whatever rebuilding the scene needs (RoboCasa: model xml, states)
    expert.npz         the demonstrator record: verification, audit, the replay stub, recipes after close
```

| Where | Public? | May hold a demonstrator record? | Read by |
|---|---|---|---|
| `demo_frames.npz` | yes | **no** | the fork, to send to the policy; the runtime |
| `demo.json` | only after close; never sent to a policy | **no** | the fork (to build `info`), the competition, the dashboard (closed epochs only) |
| `demo.mp4` | yes | no (it is a preview of the frames) | people |
| `private/scene.json`, `private/scene/` | no | yes (RoboCasa simulator states) | the fork, to rebuild and verify the scene |
| `private/expert.npz` | no | **yes: this is where it goes** | see §8 |

After an epoch closes, nothing in a bundle is privileged, and whole bundles MAY be published.

**The manifest's cameras (Q13, Q4), as `bundle.write` and `bundle.read` check them**

| `demo_source` | `camera` holds exactly |
|---|---|
| `expert`, `mimicgen` | `name`, `w`, `h` (pixels), `fovy` (degrees), `pose` = `[x, y, z, qw, qx, qy, qz]`: metres and a unit quaternion (within 1e-3), in the frame the bundle's `action_spec.frame` names |
| `humangen` | `name`, `w`, `h`, `source`, `video` (non-empty strings: where the video comes from, and which video it is) |

- `camera.name` names a `frames_<camera.name>` array: it is the primary camera.
- `camera.w` and `camera.h` are that array's own width and height, as `info.cameras`' are an observation's (Q3 C-P3): a lens no frame was rendered through says nothing about the demonstration.
- `cameras` lists every demonstration camera, `camera.name` first and the rest in ascending name order, and names exactly the `frames_` arrays. `info.demo_cameras` is sent equal to it (§5).
- Any other `camera` field (a lens `type`, say) is refused: the fields are Q13's, per source.
- No file of a bundle is a symlink, `demo.json` included: it is the file the digest is taken of, so a link would leave the manifest swappable with every hash still matching. `bundle.read` refuses one (`BundleError`).

---

## 3. The allow-list

```python
# zerowam_protocol/bundle.py
PUBLIC_PREFIXES = ("frames_",)
PUBLIC_NAMES = ("times",)
```

| Where it is checked | What happens to any other name |
|---|---|
| `bundle.write` | `BundleSchemaError` (a subclass of `BundleError`), naming the array and Q4 |
| `bundle.read` | `BundleSchemaError` |
| `RemotePolicy.set_demonstration` | `BundleSchemaError` (a `ValueError`), before anything is sent. Never `PolicyUnavailable`, so it is never scored against a submission. |

**Exit codes in a fork**

| Error | Exit | Meaning (plan §4.3) |
|---|---|---|
| `BundleSchemaError`, at write, read or send | **2** | Invalid input (config, bundle schema): stop and fix. It is the fork's own bug, so a rebuild or a retry would repeat it. |
| Any other `BundleError` (a hash that does not match; a missing file or one that is not the bundle's; a `demo.json` or an npz that cannot be read) | 4 | Bundle verification failed: rebuild the unit; void it if the failure repeats. |

A fork catches `BundleSchemaError` before `BundleError`. A `BundleSchemaError` raised by `set_demonstration` inside the rollout MUST pass the rollout's catch-all and exit 2. It is never recorded as a harness void or as Q6's exit 3.

**Frames and times**

| Name | dtype | Shape | Rules |
|---|---|---|---|
| `frames_<camera>` | uint8 | (T, H, W, 3), RGB, row 0 at the top | One per name in `info.demo_cameras`. Native resolution and native recording rate: never resampled or resized by the benchmark. All share T, and T ≥ 2. |
| `times` | float64 | (T,) | Seconds. Finite and never decreasing. `frames_*[t]` was recorded at `times[t]`. `times[0]` need not be 0. |

---

## 4. What `set_demonstration` carries

```python
from zerowam_protocol import bundle

manifest, arrays = bundle.read(bundle_dir)  # refuses a bundle with a disallowed array
policy.set_demonstration(bundle.public_arrays(arrays), info)
```

- **`arrays`:** exactly `bundle.public_arrays(arrays)`. That is `frames_<camera>` for each demonstration camera, and `times`.
- **`info`:** the keys Q14 defines. Only the three keys in §5 come from the demonstration. The rest describe the evaluation:
  - `embodiment`;
  - `action_spec` (Q3);
  - `cameras` (the observation cameras, Q3);
  - `step_limit`;
  - …
- **Other messages.** `reset` carries only the policy seed. `act` carries only the evaluated robot's own observation (§1.1): `frames_<name>` for each `name` in `info.cameras`, the array `info.action_spec.state_channel` names (`endpose`), and the native `qpos`, with layouts per Q3.
- **Never.** No message carries `private/`, `demo.json`'s `scene_seed` or fingerprint (Q5), or any demonstrator record.

---

## 5. `info` keys that come from the demonstration

| Key | Type | Rule |
|---|---|---|
| `demo_cameras` | list[str], non-empty | Camera names, without the `frames_` prefix. Element 0 is the primary camera, the manifest's `camera.name` (Q13); the rest follow in ascending name order. The list equals the set of `frames_` arrays, and the manifest's `cameras` list (`demo.json`) is identical, in the same order. Do not confuse it with `info.cameras`, which lists the evaluated robot's observation cameras as `{name, role, w, h}` (Q3): on HumanGen, `demo_cameras` is `["human"]` while `info.cameras` names the robot's cameras. Required. |
| `instruction` | str | Exactly `"Follow the demonstrated behavior."`, on every axis. Required. Never task metadata. |
| `demo_text` | non-empty str | Present if and only if the demonstration's own source supplies a caption of that exact video. Today that is only `demo_source == "humangen"` with `humangen.caption: true`, and the value is the pairing file's `human_task_name` for that video. When absent, the key is omitted: never `""` and never `null`. A benchmark never writes a caption of its own, because that would be task language. |

---

## 6. Per axis

| Axis | `demo_source` | Public arrays | `demo_cameras` | `demo_text` | Evaluation scene | `private/expert.npz` holds | Observation keys each step (Q3 layouts) |
|---|---|---|---|---|---|---|---|
| robotwin_humangen | humangen | `frames_human` (T, 828, 1108, 3) at 24 fps, `times` | `["human"]` | yes, with a caption config (Q1) | `new_seed` | the evaluation scene's expert, which is **not** the demonstrator: its rows do not line up with `frames_human` | `frames_head_camera`, `frames_left_camera`, `frames_right_camera`, `qpos`, `endpose` |
| robotwin_sim | expert | `frames_head_camera` (T, 480, 640, 3; Large_D435); `frames_left_camera` and `frames_right_camera` (T, 240, 320, 3); one frame every 15 control steps; `times` | `["head_camera", "left_camera", "right_camera"]` | no | `same_as_demo` | `qpos`, `endpose`, `actions`, `ee_actions`, `times` | same as robotwin_humangen; the head camera is observed as a D435 at 320×240 |
| robocasa_sim | mimicgen | `frames_robot0_head` (T, 480, 640, 3) at 20 Hz, `times` = t/20 | `["robot0_head"]` | no | `same_as_demo` | `states`, `endpose`, `actions`, `ee_actions`, `times`; plus `private/scene/` | `frames_robot0_head`, `frames_robot0_eye_in_hand`, `frames_robot0_agentview_right` (256×256), `qpos`, `endpose` |

Each fork pins its axes' public key sets and observation key sets in tests.

---

## 7. The demonstrator record: `private/expert.npz`

**Required arrays** (for a demonstrator with a state: `expert`, `mimicgen`)

| Array | Shape | Meaning |
|---|---|---|
| `endpose` | (T, A) | Row t is the demonstrator's state at public frame t. |
| `ee_actions` | (T−1, A) | `ee_actions[t]` is the `ee` action that takes frame t to frame t+1. A = `info.action_spec.action_dim`. `ReplayPolicy` plays this array. |
| `times` | (T,) | Equal to the public `times`. |

**Optional arrays:**
- `qpos` (native to the benchmark);
- `actions` (the benchmark's native action);
- `states` (simulator state);
- finer-rate arrays, each with its own `<name>_times`.

**Layouts**
- `endpose` and `ee_actions` MUST come from the same functions that produce the evaluated robot's observation state channel and interpret its executed `ee` action. They therefore follow Q3 (§1.1): per arm `[x, y, z, qw, qx, qy, qz, g]`, in the frame the `action_spec` recorded in `demo.json` names, with g = 1 fully open.
- `qpos`, `actions` and `states` stay native.

**HumanGen bundles.** In a `humangen` bundle, `expert.npz` holds the evaluation scene's expert, with that expert run's own `times`. Nothing may treat it as the trajectory of `frames_human`.

---

## 8. Who may read `private/`

| Reader | May read | Rule |
|---|---|---|
| The fork's `eval run`, serving a submission | `private/scene.json`, `private/scene/` | MUST NOT open `private/expert.npz`. A fork test proves it: an `expert.npz` made unloadable as data (see below) gives the same result. |
| **`ReplayPolicy`** (`zerowam_protocol.stubs`) | `private/expert.npz`, key `ee_actions`, handed to it by the harness | **The one exception.** A test instrument for the T1 selftest and `--stub-policy replay`. It MUST NOT be served in a scored epoch. A stub run records `stub_policy` in `result.json`, and the competition refuses to score such a result outside a dry run. |
| `demo verify`, audit | everything | organiser tools only |
| Recipes shipped with a runtime | `private/expert.npz`, and `private/scene/` to re-render views | only on bundles of closed epochs, or on the participant's own `demo make` output |
| Any policy or runtime being served | nothing | — |

**`bundle.read` and `expert.npz`.** Since bundle version 2, `bundle.read` verifies every file under `private/` against `private_files`: it reads `expert.npz`'s bytes into a sha256 and never parses or returns them. Reading a bundle is therefore not "opening `expert.npz`" in the sense of the rule above, and the fork test that proves the rule makes `expert.npz` unloadable as data, not unreadable as bytes: it makes every attempt to load that path raise (for example by wrapping `numpy.load`) and checks that the result is unchanged. A file whose bytes cannot be read fails `bundle.read` itself, as any damaged bundle does (exit 4).

---

## 9. How a family declares its inputs

The vocabulary is defined here, beside the allow-list, so that it binds every family whichever runtime repository ships it:

```python
# zerowam_protocol/bundle.py
DEMONSTRATION_INPUTS = ("video", "caption")
PROMPT_LANGUAGES = ("none", "generic", "demonstration_caption")


def check_demonstration_inputs(inputs: Mapping[str, Any]) -> list[str]:
    """Every problem with a family file's `inputs` block (empty when it is valid)."""
```

A family file declares:

```yaml
inputs:
  demonstration: [video, caption]      # MUST contain video; MAY contain caption; nothing else
  prompt_language: demonstration_caption
```

**`inputs.demonstration`**

| Value | The runtime reads | Guaranteed by benchmarks |
|---|---|---|
| `video` (required) | `frames_<info.demo_cameras[0]>` and `times`. It MAY read other cameras in `demo_cameras`, if its family file names them per embodiment. | every axis |
| `caption` (optional) | `info.demo_text`, when present | HumanGen with a caption only. The runtime MUST work without it. |
| anything else, including `proprio` and `actions` | refused, with a message citing Q4 | never provided |

**`inputs.prompt_language`**

| Value | Text the family conditions on |
|---|---|
| `none` | no text |
| `generic` | `info.instruction` |
| `demonstration_caption` | `info.demo_text` when present, otherwise `info.instruction`. It requires `caption`. |
| anything else, including `task` | refused: no benchmark sends task language |

**Who calls the check**
- The competition's `config check` (C3) reads the family file from the runtime checkout that the competition config names, and calls `check_demonstration_inputs` on it without importing that runtime. Naming that checkout is a new config entry, added by C3: today `competition.yml` names only the family.
- Each runtime's own family loader also calls it. `zerowam-runtime` imports the constants; it does not define its own.

**Reading rule.** A runtime MUST NOT read any demonstration array or demonstration `info` key beyond what its declared inputs name. A runtime test proves this with a demonstration that carries extra arrays and keys.

---

## 10. Adding a benchmark (a fork)

1. Record every demonstration camera as `frames_<camera>`: uint8 RGB, at native resolution and native rate. Record `times` in seconds.
2. Set `camera.name` (Q13) to the primary camera. Send `info.demo_cameras` with the primary camera first and the rest in name order, and make the manifest's `cameras` identical.
3. Send `info.instruction = "Follow the demonstrated behavior."`. Send `demo_text` only if the video's own source supplies its caption.
4. Write the demonstrator record, if you have one, to `private/expert.npz`, with the required arrays of §7, by passing it to `bundle.write(..., expert=arrays)`. Write anything else under `private/` (a scene) before calling `bundle.write`: it hashes every private file, and `bundle.read` refuses one added later. Produce `endpose` and `ee_actions` with the functions that serve your observation and execute your action (Q3).
5. Build `set_demonstration`'s arrays with `bundle.public_arrays`, and nothing else.
6. Make per-step observations hold only the evaluated robot's own channels: `frames_<name>` for each `name` in `info.cameras`, the array `info.action_spec.state_channel` names, and the native `qpos` (§1.1).
7. Map `BundleSchemaError` to exit 2, including one that `set_demonstration` raises inside your rollout (let it pass your catch-all). Map other `BundleError`s to exit 4.
8. Tests, using P10's conformance helpers:
   - the public key set per axis;
   - the observation key set per axis;
   - `eval run` never opens `expert.npz`;
   - the T1 replay selftest ends in `success`.
9. If your demonstrator has no state (like HumanGen), write no record for it. If your harness needs a replay trajectory, it is the evaluation scene's expert, and it is never aligned with the demonstration.

## 11. Adding a model family (a runtime)

1. Declare `inputs.demonstration` and `inputs.prompt_language` from §9's vocabulary. Run `check_demonstration_inputs` in your family loader; the competition runs it too.
2. Read only `frames_<info.demo_cameras[0]>` (plus any cameras your family file names), `times`, and the text your `prompt_language` names.
3. Resample by `times` yourself. Bundles are never resampled for you.
4. Work without `demo_text`: it is absent on every axis but a captioned HumanGen axis.
5. Do not expect `qpos`, `endpose` or actions in the demonstration. They are never there. The robot's own state arrives in each observation, per Q3.
6. If you ship training recipes:
   - read targets from `private/expert.npz` of closed epochs' bundles, or of your own `demo make` output, never from public arrays;
   - re-render observation views from `private/` where the benchmark allows it (RoboCasa), or mark your output as limited to the demonstration views.

## 12. A future family that wants more than video

**A demonstrator record (`proprio`, `actions`).** You cannot opt in from your family file: the vocabulary refuses it. You need a new decision in the competition's `decisions.md` that supersedes Q4-2 and Q4-5, and that decision MUST:
1. use a **new axis with `evaluation.scene: new_seed`** whose source records a demonstrator state, with its own score series.
   - It is never a `same_as_demo` axis: there, the record replays the evaluated scene.
   - It is never HumanGen, which records no demonstrator state. Its paired robot episodes are another scene, and are excluded for that reason.
2. **measure a replay floor** first: `ReplayPolicy` plays the demonstration's `ee_actions` on the axis's `new_seed` evaluation scenes. If replay succeeds on more than a fraction *f* stated in advance, the axis is refused, unless the decision keeps it with that floor published as its baseline beside every score.
3. change this package:
   - bump `BUNDLE_VERSION`;
   - record `eval_scene` in the manifest;
   - add public names with a `demo_` prefix (`demo_qpos`, `demo_endpose`, `demo_ee_actions`), accepted by `bundle.write` only when `eval_scene == "new_seed"`;
   - extend `DEMONSTRATION_INPUTS` as the decision names.
4. keep the family runnable without those inputs, because the gate axis never carries them.
5. make the competition's config check refuse an epoch whose axes and family disagree. A conformance test proves that a `same_as_demo` bundle carrying a record is refused.

**A depth rendering of a demonstration camera.** This route is lighter and is not tied to `new_seed`. It needs:
- a decision naming the channel and the axes;
- a public prefix such as `depth_<camera>`, each array matching a `frames_<camera>` with the same T and `times`;
- a vocabulary value such as `depth`;
- a `BUNDLE_VERSION` bump, and conformance checks.

Segmentation masks and point clouds carry simulator object ids or camera extrinsics, so they need their own analysis.

Zero-WAM takes part in neither route under D1 and the current architecture: it has no input for either.

---

## 13. Mistakes this prevents

| Mistake | What stops it | Where |
|---|---|---|
| A fork makes `qpos`, `endpose` or `actions` public on a `same_as_demo` axis. That gives every policy the replay trajectory for the scene it is scored in. | The allow-list refuses them at write, at read and at send | `bundle.write`, `bundle.read`, `RemotePolicy.set_demonstration` |
| A fork adds the record under a new name (`demo_joints`, `gripper_track`) | The allow-list is closed. Only `frames_*` and `times` pass. | same |
| A fork puts the record in `demo.json` or `info` | P11's `info` schema and P10's `check_info` fix the demonstration keys and their values; review of the fork's `info` builder against Q14's key list catches the rest. `demo.json` is never sent and never leaves before close. | P10, P11, Q4-8 |
| A fork smuggles the record into per-step observations | Fork tests pin each axis's observation key set to the evaluated robot's own channels | RT16, RC11 |
| The harness's evaluation reads `expert.npz` while serving a submission | A fork test with an `expert.npz` made unloadable as data (§8) | RT16, RC11 |
| `ReplayPolicy` is served as if it were a submission | A stub run records `stub_policy`, and scoring refuses it outside a dry run | RT9, C10 |
| A new family declares `proprio` in its own family file and ships a check that allows it | The vocabulary lives here, and the competition checks every family file against it | P14, C3 |
| A runtime reads a key it did not declare (an old array, a caption it said it would not use) | A runtime test with extra arrays and keys | runtime tests (RU9) |
| A disallowed array is treated as a hash failure (rebuilt, then voided for everyone), or at send as a harness void or a retry | `BundleSchemaError`, at write, read or send, exits 2 and stops the build | forks (RT3, RC1), competition `epoch.py` |
| A recipe trains on a live epoch's expert trajectories because they happened to be public | Shipped recipes read only `private/expert.npz`, only from closed epochs or from `demo make` output | recipes (RU15) |
| A recipe pairs a HumanGen bundle's `expert.npz` with `frames_human` as if it were that video's trajectory | The rule in §7, and the recipe gives HumanGen bundles no robot target | RU15 |
| A benchmark writes its own caption for a robot demonstration, adding task language | `demo_text` only from the video's own source; conformance refuses it elsewhere | P10 |
| Someone "fixes" Zero-WAM by feeding it the robot's joints | Zero-WAM has no input for them. Its family declares `[video, caption]`, and the vocabulary refuses anything else. | P14, families/zerowam.yml |
