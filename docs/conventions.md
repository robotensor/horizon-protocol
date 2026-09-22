# Action and observation conventions

Normative source: robotensor/vicl-competition docs/decisions/q3-conventions.md (decision Q3); where they differ, that record wins.

This is the specification for anyone who adds a **new benchmark** (or a new robot to an existing
one) or a **new model family** to the competition. It says what every number exchanged between a
benchmark and a model runtime means, who converts it, who maps it, and how both sides prove they got
it right. You do not need to have read the discussions behind it.

**Words.** MUST, MUST NOT, SHOULD and MAY are normative.
**Evidence tags.** [code] = the code shows it; [data] = read from released data; [measured] = a check
run on CPU; [inference] = reasoning, not shown by code.
**Paths.** Line numbers are at these commits. `RoboTwin/…` is the RoboTwin fork (`robotensor/RoboTwin-VICL`,
branch `zerowam`, `ffeaed6`; `robot.py` = `envs/robot/robot.py`, `_base_task.py` =
`envs/_base_task.py`); `robocasa/…` is the robocasa fork (branch `zerowam`, `f116b56`);
`robosuite/…` is robosuite `5ce6643f`; `Zero-WAM/…` is Zero-WAM `08e2c4a`; `runtime/…` is
`vicl-runtime-zerowam` `a42edf7`; this package's own files are at `c0bb625`. `transforms3d
quaternions.py` is `transforms3d/quaternions.py` of the transforms3d package RoboTwin runs with
(0.4.2). The constants and the pure checks C-P1..3 named below live in
`vicl_protocol.conventions`; the helpers the forks' selftests use live in
`vicl_protocol.conformance` (P10).

**Status.** This page is the contract; where an item below has not landed, the page is what
implementers follow.

- **Landed in this package:** `vicl_protocol.conventions` (P9): the constants, `check_action_spec`
  (C-P1), `check_chunk(a, spec)` (C-P2), `check_observation(obs, spec, cameras)` (C-P3),
  `arm_slices` and `same_rotation`. `demo.json`'s `action_spec`, required and checked with
  `check_action_spec` by `bundle.write` and `bundle.read`, with `BUNDLE_VERSION` 2 (P2). The `info`
  schema, `vicl_protocol.info` (P11): `REQUIRED_KEYS` and `check_info`, run by
  `RemotePolicy.set_demonstration` before it sends and by the server on every `prompt`.
  `PROTOCOL_VERSION` is 3 (P6): `hello` declares each end's protocol, the action types the
  benchmark executes and whether it honours `observe_every`, and its reply may carry `served`.
- **Landed in this package (continued):** `vicl_protocol.conformance` (P10), the suite the
  forks' selftests, the runtime's tests and the harness run: `check_action_spec` (C-P1 as one
  exception type, with the arms' slices), `hold_still` (a row of the declared layout, through C-P2
  and C-P3), `check_policy` and `check_served` (a policy built through `serve.build_policy` and
  driven, in one process and over the socket; with `repeat=True`, twice from one seed for the same
  answers), `check_bundle`, `check_result`, and the `demonstration` and `observation` they send. It
  is in the wheel and needs numpy alone. The consumers' adoption is theirs: RT16, RC11 (the forks)
  and RU9 (the runtime).
- **Pending elsewhere:** the forks' side is RT5, RT10, RT11 (RoboTwin) and RC4, RC5, RC9
  (robocasa); the runtime's is RU5, RU8, RU9, RU10 and G7; the competition's check is C3. Issues:
  the "Code shape" milestone of each repository.

---

## 1. Three layers

| Layer | Unified? | Who converts or maps | Where it lives |
|---|---|---|---|
| **Wire**: ops `hello/reset/prompt/act/close`, named arrays and JSON fields | Yes | — | this package (`wire.py`, `serve.py`, `client.py`) |
| **Conventions**: what each number means: units, quaternion order and meaning, gripper range, direction and command semantics, per-arm layout, arm order, and the vocabulary of frame, tool-axis and camera-role names | Yes | **Each benchmark fork converts** its simulator's values to these, in both directions | defined and checked here; converted in the fork |
| **Action/observation space**: which arms and cameras exist, which frame poses are in, the tool point and axes, how an action executes, what the observed gripper value is, held DoFs | No: native per robot | **The fork declares it** in `info.action_spec`, `info.cameras` and `demo.json` `action_spec`. Nobody converts it | the fork |
| **Model layout**: channels, slots, relative or absolute, quaternion arithmetic, normalisation, view order and size, observation cadence | No: per model family | **The family's runtime maps**, per embodiment, through one table that pins the spec it serves | that family's runtime repo only |

A benchmark never learns a model's layout. A model never learns a simulator's quirks. Anything that
cannot be converted without undoing what a model was trained on (the frame) or cannot be converted
at all (whether the gripper reading is commanded or measured) is declared, never guessed.

---

## 2. Mistakes this prevents (read this before writing code)

Each item is a trap that was found in this code base. The rule that prevents it is normative.

1. **A wire arm name is not a model slot.**
   A one-armed robot is declared `arms: ["right"]` on the wire, but Zero-WAM puts a single arm in its
   **slot 0: channels 0-6 (pose) and 28 (gripper)**, with every other channel masked. Its loader
   writes each feature from the start of its group, and its test asserts mask[28] true and mask[29]
   false (Zero-WAM/wan_va/dataset/lerobot_action.py:199-247; Zero-WAM/tests/test_lerobot_action.py:228-230)
   [code]. The runtime once put a single arm in the right slot with the left "held", claiming that
   was the training layout (runtime/src/vicl_runtime_zerowam/actions.py:21-22); it was not.
   **Rule:** a runtime maps by its embodiment table only and MUST NOT derive a slot from an arm name.
2. **RoboTwin's world frame is not a base frame.**
   RoboTwin reports and executes end-effector poses in the **world** frame (RoboTwin/robot.py:595-602;
   RoboTwin/envs/robot/planner.py:100-105) [code]. Its robot base sits at world (0, −0.65, 0), turned
   90° about z (RoboTwin/assets/embodiments/aloha-agilex/config.yml:20) [code]. Zero-WAM's checkpoint
   learned world-frame values [data].
   **Rule:** the frame is declared (`action_spec.frame`), with each arm's base pose in `base_poses`
   when it is `world`. Forks never convert frames; a runtime that needs another frame converts from
   `base_poses`.
3. **RoboTwin's quaternion is wxyz, but Zero-WAM's RoboTwin arithmetic reads it as xyzw.**
   RoboTwin's endpose quaternion comes from transforms3d `mat2quat`, which returns (w, x, y, z) with
   w ≥ 0 (RoboTwin/robot.py:597-602; transforms3d quaternions.py:213-217) [code]. Zero-WAM's RoboTwin
   training and client pass those raw numbers to scipy `Rotation.from_quat`, which reads them as
   (x, y, z, w) (Zero-WAM/wan_va/dataset/robotwin_action.py:26-31;
   Zero-WAM/evaluation/robotwin/eval_policy_client_openpi.py:505-515) [code]. This is a
   **RoboTwin-only compatibility rule** of the Zero-WAM runtime, pinned as `robotwin_legacy`.
   **Rule:** the wire is wxyz. Do not "fix" the RoboTwin arithmetic (it would break parity with the
   released model), and do not copy it to a new embodiment (new embodiments use the true reading).
4. **Gripper 1 = open.**
   An early plan table said "[0 open, 1 closed]". RoboTwin, and the checkpoint trained on it, use
   **0 = closed, 1 = open** (RoboTwin/_base_task.py:1384-1388; RoboTwin/robot.py:538-554) [code].
   **Rule:** `g` is in [0, 1], 0 fully closed, 1 fully open, in actions and observations.
5. **robosuite quaternions are xyzw, and float32.**
   robosuite's transform utilities use (x, y, z, w) (robosuite/robosuite/utils/transform_utils.py:4)
   and `mat2quat` casts to float32 (transform_utils.py:327) [code].
   **Rule:** a robosuite-based fork computes the wire quaternion itself, in float64, in wxyz order.
6. **RoboCasa's native gripper is sign-based and reversed.**
   robosuite's Panda gripper takes −1 = open, +1 = closed, and acts on the **sign** only: each step
   moves the finger target by 0.2 · sign(action) (robosuite/robosuite/models/grippers/panda_gripper.py:43-62)
   [code]. The Panda starts half open (g ≈ 0.52, panda_gripper.py:25-26) [code], so passing a wire
   value through as a direction would make "echo the state" open the gripper to its end stop.
   **Rule:** `gripper_command` is `"position"`: the fork converts to a position target (§8).
7. **Echoing the state holds still; a zero action does not.**
   Actions are absolute targets, so an all-zero action asks an arm to reach the frame origin with a
   zero (invalid) quaternion [inference]. The stub "hold still" policy sent exactly that until P8.
   **Rule:** sending the observed state channel back as the action MUST hold the robot still,
   gripper included (C-F1). A "do nothing" policy echoes the state: `stubs.ZeroPolicy` echoes the
   array `info.action_spec.state_channel` names, and refuses a demonstration that declares none
   rather than command a pose of zeros.
8. **`qpos` is benchmark-native and outside the convention.**
   RoboTwin's `qpos` is 6 joint drive targets plus a normalised gripper per arm
   (RoboTwin/robot.py:494-506); RoboCasa's is the 7 arm joints without the gripper
   (robocasa/robotensor_bench/env.py:388-389) [code].
   **Rule:** no model family may consume `qpos` until a later decision defines it; a fork MUST
   refuse a policy that declares `qpos` actions.
9. **`action_type`, `action_dim` and `control_hz` exist only inside `action_spec`.**
   The forks used to disagree: one sent `action_type`/`action_dim`/`action_dims` at the top level and
   no `control_hz`; the other spread its whole spec over the top level
   (RoboTwin/robotensor_bench/evaluate.py:79-90; robocasa/robotensor_bench/evaluate.py:78-88) [code].
   **Rule:** the top-level keys `action_type`, `action_dim`, `action_dims` and `control_hz` MUST NOT
   be sent; the protocol refuses them.
10. **Tool axes differ between robots.**
    RoboTwin's endpose describes the link6 flange and approaches along its **+x** axis
    (RoboTwin/robot.py:557-561, 597-602) [code]; robosuite's Panda eef describes the grip site and
    approaches along **+z** (robosuite/robosuite/models/assets/grippers/panda_gripper.xml:20-24) [code].
    The same quaternion points the two grippers differently.
    **Rule:** the fork declares `tool: {point, approach_axis, closing_axis}`; a family that depends on
    tool axes reads them and pins them.
11. **"Robot base" on a mobile manipulator is the arm's base link, not the mobile base.**
    robosuite's arm controller works in site `robot0_right_center` on Panda link0, which sits on the
    torso lift (robosuite/robosuite/controllers/composite/composite_controller.py:142-153;
    robosuite/robosuite/models/assets/robots/panda/robot.xml:138-139) [code].
    **Rule:** `robot_base` means the base link of the arm's kinematic chain, and is allowed only when
    nothing below it moves during the episode.
12. **Do not type fixtures from a document.**
    RoboTwin's base pose is configured as `[0, −0.65, 0, 0.707, 0, 0, 0.707]` and kept in float32,
    unnormalised: its quaternion norm is 0.999849, not 1, and −0.65 reads back as −0.64999998
    [measured]. A fixture typed as 0.7071 is off by 1e-4.
    **Rule:** pinned specs and fixtures are regenerated from the fork's spec function; `base_poses`
    compare at 1e-5; a consumer normalises their quaternions.
13. **Quaternions have two signs.**
    q and −q are the same rotation. Zero-WAM's training transforms flip the relative quaternion so its
    last component is positive (Zero-WAM/wan_va/dataset/robotwin_action.py:29-31;
    Zero-WAM/wan_va/dataset/lerobot_action.py:64-69) [code]; code that does not will differ on about
    half of random inputs while being correct.
    **Rule:** every consumer accepts either sign, and every test compares quaternions up to sign.
14. **Observation cameras are not demonstration channels.**
    On RoboTwin's HumanGen axis, `info.cameras` names the human video channel (`["human"]`) instead
    of the robot's cameras, because it is copied from the bundle's demonstration channels
    (RoboTwin/robotensor_bench/demo.py:201-203; RoboTwin/robotensor_bench/evaluate.py:80) [code].
    **Rule:** `info.cameras` lists observation cameras with roles; the demonstration's channels go in
    `info.demo_cameras`.
15. **A waypoint is not a control step.**
    A RoboTwin action is a target the benchmark plans to and reaches before the next one, taking a
    variable number of simulator steps (RoboTwin/_base_task.py:1574-1592, 1628-1662) [code]; a
    RoboCasa action is one controller step at 20 Hz.
    **Rule:** read `action_spec.execution` and `control_hz`; never assume a rate.
16. **The observed gripper value may be commanded or measured.**
    RoboTwin reports the last commanded value (RoboTwin/robot.py:526-536) [code]; RoboCasa reports the
    measured finger opening (§8).
    **Rule:** read `action_spec.gripper_state`.

---

## 3. Wire conventions (unified; each fork converts)

| Quantity | Convention |
|---|---|
| Length | metres |
| Angle | radians (joint angles only; no Euler angles or axis-angle cross the wire) |
| Time | seconds (`times`, float64) |
| Position | The tool point's coordinates in the frame named by `action_spec.frame` |
| Orientation | Unit quaternion, order **`(qw, qx, qy, qz)`**, Hamilton product, active: it rotates the declared frame into the tool frame declared by `action_spec.tool`, so `v_frame = R · v_tool` and the columns of R are the tool axes expressed in the frame. **Either sign is valid.** Producers SHOULD send `qw ≥ 0`; every consumer MUST accept either sign. |
| Quaternion norm | Producers MUST send ‖q‖ within 1e-3 of 1. Before executing, a fork MUST normalise, and MUST refuse a quaternion that is non-finite or has ‖q‖ < 1e-6 (a refused policy output is scored as a `failure`). |
| Gripper `g` | **[0, 1], 0 = fully closed, 1 = fully open**, in actions and observations. In an action, `g` is a **position target**: the fork drives the gripper toward that opening and holds it there; a target equal to the observed `g` holds still. The fork MUST clip `g` to [0, 1], not refuse it. |
| End effector | One gripper degree of freedom per arm. End effectors with more (for example dexterous hands) are out of scope and need a new decision. |
| Arm order | Two arms: `left` then `right`. One arm: a single block, declared `arms: ["right"]`. The name is a label; **it never decides a model slot.** |
| EE action | Absolute target (never a delta), `action_type = "ee"`. One action is `(A,)`; a chunk is `(H, A)`, row 0 executed first. `A = action_spec.action_dim = 8 × len(arms)`. |
| Joint action (`qpos`) | Deferred. A fork MUST refuse a policy whose `hello` declares `qpos`. |
| Images | `uint8` RGB `(H, W, 3)`, row 0 at the top (upright), at the native resolution declared in `info.cameras`. Stacked observations are `(K, H, W, 3)`. |
| Camera roles | `ego` (head or robot-mounted main view), `wrist_left`, `wrist_right` (a one-armed robot's wrist is `wrist_right`), `third` (fixed or external). |

---

## 4. The per-arm block

The per-arm layout is the constant `["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]`.

| Index, two arms (A = 16) | Index, one arm (A = 8) | Value |
|---|---|---|
| 0-2 | 0-2 | Two arms: `left` x, y, z. One arm: the single arm (declared `right`) x, y, z. In metres, in `frame` |
| 3-6 | 3-6 | qw, qx, qy, qz |
| 7 | 7 | g, 0 closed … 1 open |
| 8-10 | — | `right` x, y, z |
| 11-14 | — | `right` qw, qx, qy, qz |
| 15 | — | `right` g |

The observation state channel (named by `action_spec.state_channel`) uses exactly this layout, `(A,)`
or `(K, A)` when stacked. **Sending it back as the action (its last row, when stacked) MUST hold the
robot still, gripper included.**

---

## 5. `info.action_spec`

The one object that describes a benchmark's action space. The fork sends it in `info` with the
demonstration and records it unchanged in `demo.json` under `action_spec`. Values are shown for
reading; fixtures MUST be regenerated from the fork (C-R1).

| Field | Type | Rule | aloha-agilex (RoboTwin) | PandaOmron (RoboCasa) |
|---|---|---|---|---|
| `action_type` | str | `"ee"` (`qpos` is deferred) | `"ee"` | `"ee"` |
| `action_dim` | int | `8 × len(arms)` | 16 | 8 |
| `arms` | list[str] | `["left","right"]` (that order) or `["right"]` | `["left","right"]` | `["right"]` |
| `layout` | list[str] | The constant per-arm layout of §4 | constant | constant |
| `frame` | str | `"world"`, or `"robot_base"` = the base link of the arm's kinematic chain, the frame the arm's own controller or planner works in. `robot_base` is allowed only when every declared arm shares that one base link and no DoF below it moves during the episode. | `"world"` | `"robot_base"`: robosuite site `robot0_right_center` on Panda link0 (on the torso lift, not the mobile base; base and torso held) |
| `base_poses` | {arm: [x,y,z,qw,qx,qy,qz]} | Required when `frame` is `"world"`, absent otherwise: each arm's base-link pose in the world, **as the simulator reports it** (not normalised, not rounded). Same form and quaternion order as a tool pose: with `t = [x,y,z]` and `R = R(qw,qx,qy,qz)`, `v_world = R · v_base + t` (RoboTwin/robot.py:570-577 inverts exactly this) [code]. Compared at 1e-5; normalise the quaternion before use. | both `[0, -0.65, 0, 0.707, 0, 0, 0.707]` (rounded here), read live from `env.robot.left_entity_origion_pose` / `right_entity_origion_pose` (RoboTwin/robot.py:63-65, 90-93) [code] | absent |
| `tool` | {point, approach_axis, closing_axis} | `point`: text naming the point the position describes. `approach_axis`: `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"` or `"-z"`, the tool-frame axis from the palm toward the fingertips. `closing_axis`: `"x"`, `"y"` or `"z"`, the tool-frame axis the fingers move along (unsigned). | `{point: "RoboTwin endpose: the fl_joint6 / fr_joint6 frame on link6 (the flange), offset gripper_bias − 0.12 = 0 m along +x", approach_axis: "+x", closing_axis: "y"}` (RoboTwin/robot.py:557-561, 597-602) [code + inference from the URDF finger joints] | `{point: "robosuite grip_site: the eef body, 0.097 m from the hand root along +z", approach_axis: "+z", closing_axis: "x"}` (robosuite panda_gripper.xml:14-30, 38-40; robosuite/robosuite/robots/robot.py:929) [code + inference from the xml] |
| `execution` | str | `"waypoint"`: each action is a target the benchmark plans to and reaches (or gives up on) before the next. `"setpoint"`: each action is the controller target for one step of 1/`control_hz` s. | `"waypoint"` | `"setpoint"` |
| `control_hz` | number or null | Required for `setpoint`; `null` for `waypoint` | `null` | 20 |
| `gripper_command` | str | `"position"`: `g` is the target opening; a target equal to the observed `g` holds still. The only value defined; a simulator commanded otherwise MUST be converted by its fork. | `"position"` | `"position"` (converted by the fork, §8) |
| `gripper_state` | str | `"commanded"`: the observed `g` is the last commanded value. `"measured"`: the physical opening. | `"commanded"` | `"measured"` |
| `state_channel` | str | The observation array carrying the state in the §4 layout | `"endpose"` | `"endpose"` |
| `held` | list[str] | Native DoFs the fork holds fixed (informational) | `[]` | `["base","torso"]` |
| `native_action_dim` | int | Informational | 16 | 12 |

`check_action_spec(spec)` enforces: every field present and allowed; `action_dim == 8·len(arms)`;
`base_poses` present exactly when `world`, each quaternion within 1e-3 of unit norm; `control_hz` set
exactly when `setpoint`; `layout` is the constant; `gripper_command == "position"`; `tool` complete.

Every number it reads must be one a double can hold: JSON carries integers that are not (`10 ** 400`),
and a `base_poses` component or a `control_hz` out of that range is listed as a problem like any
other, never raised as an `OverflowError`. The same rule holds wherever this package range-tests a
number a fork or a file gave it: `demo.json`'s `camera.pose` and `camera.fovy`, and `result.json`'s
`timing`.

---

## 6. `info`: the keys these conventions touch

- `action_spec` (§5) is required.
- `cameras` is a list of `{name, role, w, h}`, one per **observation** camera, in the order the
  benchmark renders them. `name` MUST match an observation array `frames_<name>`.
- `demo_cameras` names the demonstration's channels. They never go in `cameras`. (`demo.json`'s own
  `cameras` list is the demonstration's channels, equal to `demo_cameras`; it is not `info.cameras`.)
- `action_type`, `action_dim`, `action_dims` and `control_hz` MUST NOT appear at the top level
  (`conventions.REFUSED_INFO_KEYS`).
- The other keys (`embodiment`, `step_limit`, `instruction`, `demo_text`, …) are defined by the
  protocol's `info` schema (vicl-competition decisions Q14 and Q4): `vicl_protocol.info`
  holds it, and `check_info(info, arrays)` refuses an `info` that breaks it with
  `BundleSchemaError`, at the client before a byte is sent and at the server on every `prompt`.
  A fork MAY add keys of its own; none of them is read here.
- No frame rate is declared per camera. Observations are taken between executed actions (how often
  is the policy's `observe_every`), so their timing is that of the actions; a demonstration's timing
  is its `times` array.

| Embodiment | camera name → role |
|---|---|
| aloha-agilex | `head_camera` → `ego`, `left_camera` → `wrist_left`, `right_camera` → `wrist_right` |
| PandaOmron | `robot0_head` → `ego`, `robot0_eye_in_hand` → `wrist_right`, `robot0_agentview_right` → `third` (and `robot0_agentview_left` → `third` when rendered) |

**What reaches a policy** (decided by vicl-competition Q4; its specification is this package's
`docs/demonstrations.md`). With the demonstration (`prompt`): only the arrays `frames_<camera>` (one
per name in `info.demo_cameras`) and `times`, plus `info`. Never the demonstrator's state or
actions (`qpos`, `endpose`, `actions`, `ee_actions`, `states`): those stay in `private/expert.npz`.
With each `act`: only the evaluated robot's own observation (§7). Nothing under `private/`, in any
message.

---

## 7. Observation arrays

| Array | Shape | Content | Convention |
|---|---|---|---|
| `frames_<name>` | `(H,W,3)`, or `(K,H,W,3)` when the policy asked for an observation every K actions | each camera in `info.cameras` | §3 Images |
| `<state_channel>` (`endpose`) | `(A,)` / `(K,A)` | current tool pose and gripper per arm | §3-§4, in `frame` |
| `qpos` | native | joint vector | **outside the convention**; no family may consume it |

Any pose array in a bundle (`endpose`, `ee_actions`) follows §3-§4, and `demo.json` records the
`action_spec` it follows. These arrays are the demonstrator's record and live only in
`private/expert.npz`; none is public (Q4). They MUST be produced by the same functions that produce
the observation state channel and execute the action (§8).

`demo.json` `camera.pose` (an `expert` or `mimicgen` demonstration's camera) is a position in
metres plus a unit quaternion `(qw, qx, qy, qz)`, written `[x, y, z, qw, qx, qy, qz]` like a
`base_poses` entry, in the frame `action_spec.frame` names (RoboTwin
`world`; RoboCasa `robot_base`, site `robot0_right_center`). Only that frame is unified: the
camera's own axis convention (which axis looks forward, which is up) is the simulator's, SAPIEN and
MuJoCo differ, and a consumer MUST NOT assume one.

---

## 8. Who converts, who maps

| Step | Owner | Rule |
|---|---|---|
| Simulator values ↔ wire (rotation form, quaternion order, float precision, gripper range, direction and command semantics, padding of held DoFs) | Fork | One pair of pure functions per embodiment, `to_wire_state` and `from_wire_action`, unit-tested and used by observing, acting, the demonstration writer and the replay alike |
| Declaring the space | Fork | One function per embodiment returns `action_spec`; evaluation and demonstration making both use it |
| Spec, chunk and observation checks | Protocol | `check_action_spec`, `check_chunk(a, spec)`, `check_observation(obs, spec, info.cameras)` (`vicl_protocol.conventions`) |
| world ↔ base conversion, if a model needs it | That model's runtime, from `base_poses` | Forks never convert frames |
| Wire ↔ model layout | Model runtime only | One embodiment table (R1) |
| Training data in the model's layout | The runtime's recipes | The same functions as serving; `action_spec` read from each bundle's `demo.json` |
| Spec equality before an epoch | Competition | `config check` refuses to open an epoch whose declared spec the family does not serve exactly |

**Example: an identity fork (RoboTwin, aloha-agilex).** Its native end-effector observation and
action already are the wire layout: world frame, wxyz with w ≥ 0, commanded gripper with 1 = open
(RoboTwin/robotensor_bench/env.py:277-297; RoboTwin/_base_task.py:1494-1521) [code]. The fork only
declares the spec.

**Example: a converting fork (RoboCasa, PandaOmron), in float64.**

State (8 values):
1. Refresh the controller first (`composite.update_state()`, then `controller.update(force=True)`),
   because it caches the pose and only a step refreshes it (robocasa/robotensor_bench/env.py:407-414)
   [code]. Then `T = controller.goal_origin_to_eef_pose()` (env.py:415) [code]; `p = T[:3,3]`.
2. `q = (qw, qx, qy, qz)` of `T[:3,:3]`, computed in float64 (not robosuite's float32, xyzw
   `mat2quat`); if qw < 0, `q = −q`.
3. `g = clip((|finger_joint1| + |finger_joint2|) / 0.08, 0, 1)`; 0.08 m is both fingers at their
   0.04 m limit (robosuite panda_gripper.xml:30, 40) [code].

Action (8 → 12 native):
1. Refuse a non-finite action or ‖q‖ < 1e-6; normalise q; `q ← −q` if qw < 0.
2. `aa = rotvec(q)` (robosuite's absolute controller reads axis-angle).
3. Gripper as a position target: with `g = clip(g, 0, 1)`, `g_meas` the opening measured (as in
   state step 3) from the simulator when this action executes, not from the observation the policy
   was sent, and `δ = 0.05`, send −1 (open) if `g > g_meas + δ`, +1 (close) if `g < g_meas − δ`, and
   0 (hold: the finger target stays) otherwise. Echoing the observed `g` therefore holds, and a
   target of 0 keeps squeezing while an object holds the fingers apart.
4. Native 12 = `[p, aa, grip, 0, 0, 0 (base), 0 (torso), −1 (arm mode)]`
   (robocasa/robotensor_bench/env.py:65-71, 438-449) [code].

Replay actions: the wire state at t+1, with MimicGen's gripper action `a` (−1 open, +1 closed) as
`g = (1 − a) / 2`.

---

## 9. Runtime obligations (every model family)

- **R1** Keep one embodiment table entry per served embodiment, with: the pinned `action_spec` (all
  fields; strings, integers and lists exactly, `base_poses` floats within 1e-5); the model
  configuration; the model channel map; the quaternion arithmetic; the normalisation statistics and
  the loader that lays them out in the model's channels; the camera order; and the observation
  cadence. The pinned spec and its fixture MUST be regenerated from the fork's spec function, never
  typed from a document.
- **R2** In `set_demonstration`, refuse: an embodiment with no entry (no silent default); any
  `action_spec` field that differs from the pinned one; an `action_type` the family does not serve.
- **R3** Map by the table only. Never derive a model slot from an arm name.
- **R4** Output wire-convention actions: width `action_dim`, unit quaternions (either sign), `g` in
  [0, 1] with 1 = open, as a position target. If the family's training data used another gripper
  polarity, the runtime converts.
- **R5** Training recipes use the same functions as serving (no second implementation).
- **R6** A frame conversion, if the family needs one, uses `base_poses`, quaternions normalised
  first. A family trained in one frame MUST pin that frame, so that it is refused, not silently
  wrong, on a benchmark that declares another.

**Example: Zero-WAM's table** (vicl-runtime-zerowam `families/zerowam.yml` `embodiments:`; its model has
30 channels: hand pose 0-13, joints 14-27, grippers 28-29, Zero-WAM/wan_va/dataset/lerobot_action.py:14-21
[code]). It shows why a slot is the table's, not the wire's.

| Embodiment | Wire block → Zero-WAM channels | Quaternion arithmetic | Other channels |
|---|---|---|---|
| aloha-agilex (`arms: ["left","right"]`, `frame: world`) | wire `left` pose (0-6) → 0-6, its `g` (7) → 28; wire `right` pose (8-14) → 7-13, its `g` (15) → 29 (Zero-WAM/wan_va/configs/va_robotwin_cfg.py:71-78) [code] | `robotwin_legacy`: the raw wxyz numbers read by scipy as xyzw, as the released model was trained (§2 item 3) | 14-27 masked |
| PandaOmron (`arms: ["right"]`, `frame: robot_base`) | the single block's pose (0-6) → 0-6, its `g` (7) → 28: **slot 0**, although the wire names the arm `right` | `scalar_last`: wxyz reordered to xyzw, the true reading | 7-27 and 29 masked |

---

## 10. Conformance checks

Quaternion comparisons are **up to sign** everywhere: two quaternions agree when
`min(‖a − b‖, ‖a + b‖)` is within tolerance; the angle between rotations is `2·acos(|w|)` of
`q_a ⊗ q_b⁻¹`.

| ID | Where | Check | Pass |
|---|---|---|---|
| C-P1 | protocol, pure | `check_action_spec` (§5) | no error |
| C-P2 | protocol, pure | `check_chunk(a, spec)`: shape `(A,)` or `(H,A)`; finite; each ‖q‖ ≥ 1e-6 | no error |
| C-P3 | protocol, pure | `check_observation(obs, spec, cameras)`, `cameras` = `info.cameras`: state channel `(A,)` or `(K,A)`; ‖q‖ within 1e-3 of 1; g in [0, 1]; every `frames_<name>` in `info.cameras` is uint8 RGB | no error |
| C-P4 | protocol, a policy | `conformance.check_policy` / `check_served`: the declared space (C-P1), the demonstration arrays (Q4) and its `info` (Q14) are refused before a policy is built or a server is started; the policy builds through `serve.build_policy`, is reset from `seed` and driven with `calls` **different** observations, and every answer passes `checked_action`, C-P2, the wire encoder and, with a cadence, `observe.check_chunk`; with `repeat`, two runs from one seed answer the same, two NaNs in the same place being the same answer; in process the policy is closed however the check ends, one `serve.build_policy` itself refuses included, and every exception its own calls raise is a `ConformanceError`; served, the server exits 0, and one that outlives its client is killed and refused | no error |
| C-P5 | protocol, the files | `conformance.check_bundle` (bundle v2, every file held to the hash `demo.json` records - `private/` included - unless `verify=False`, and a demonstration that fits in one `prompt`: at most `wire.MAX_ARRAYS` arrays, each and their sum within `wire.MAX_MESSAGE_BYTES`) and `check_result` (result v2, and the `unit_id`, digest, task config and fork commit of the bundle it says it scored) | no error |
| C-F1 | fork selftest | Hold still: from reset, echo the state channel (last row when stacked) as the action 10 times (setpoint robots: 20 steps) | drift ≤ 5 mm and ≤ 2°; \|Δg\| ≤ 0.05 |
| C-F2 | fork selftest | Translation: +2 cm along each axis of `frame`, one at a time | displacement within 30° of the command, ≥ 50 % of its length |
| C-F3 | fork selftest | Rotation and quaternion order: target `q_axis(10°) ⊗ q_now` about each axis of `frame` | `q_obs ⊗ q_now⁻¹` has its axis within 30° of the commanded axis and an angle in [5°, 15°] |
| C-F4 | fork selftest | Gripper: `g = 1` until settled, then `g = 0` (nothing grasped), then `g = 0.5` until settled | ≥ 0.8, then ≤ 0.2, then in [0.35, 0.65] |
| C-F5 | fork selftest | Replay the private expert actions (wire convention) with the replay stub policy | `success` |
| C-F6 | fork selftest | Tool axes: (a) at reset, the palm-to-finger-pad-midpoint vector and the finger-to-finger vector, in the tool frame; (b) a target 2 cm along `R·approach_axis` | (a) within 30° of `approach_axis`, and of `closing_axis` up to sign; (b) displacement within 30° of `R·approach_axis`, ≥ 50 % of its length |
| C-R1 | runtime tests | Each pinned spec equals the fork's declared spec, regenerated from the fork | exact; `base_poses` within 1e-5 |
| C-R2 | runtime tests | Round trip wire → model → wire on random poses, per embodiment | ≤ 1e-9, quaternions up to sign |
| C-R3 | runtime tests | Agreement with the family's own training transforms (Zero-WAM: `relative_robotwin_action` and the client's `add_init_pose` for RoboTwin; `lerobot_action._relative_pose(format="xyzq")` on the reordered quaternion for PandaOmron) | ≤ 1e-9, quaternions up to sign |
| C-R4 | runtime tests | Channel map: distinct values in every model channel; only the entry's channels reach the wire, in order | exact |
| C-R5 | runtime tests | `set_demonstration` refuses an unknown embodiment and a spec with any one field changed | refused |
| C-R6 | runtime tests | The runtime's output passes C-P2 for each served spec | pass |
| C-R7 | runtime tests | The normalisation statistics are laid out over all model channels, with non-zero spans exactly on the entry's channels | exact |
| C-R8 | runtime tests | The recipe and the server read the cadence from the entry | equal |
| C-C1 | competition `config check` | For each enabled axis, the fork's declared spec is among the family's served specs | exact (`base_poses` within 1e-5), else the epoch does not open |

Tolerances and magnitudes of C-F2, C-F3, C-F4 and C-F6 are starting values; a fork MAY tune them in
its selftest and records the values it uses.

---

## 11. Checklists

**A new benchmark or embodiment**
1. Write `to_wire_state` and `from_wire_action` (§8) and the `action_spec` function (§5), including
   `frame` (and `base_poses` for `world`), `tool` and `gripper_state`.
2. If the simulator's gripper is not position-commanded, convert it to position semantics.
3. Compute quaternions in float64, wxyz; never pass a library's xyzw or float32 value through.
4. Send `info.action_spec`, `info.cameras` with roles, and `info.demo_cameras`; send no top-level
   `action_type`, `action_dim`, `action_dims` or `control_hz`.
5. Record `action_spec` in `demo.json`; store private pose arrays in the wire convention.
6. Refuse a policy that declares `qpos`.
7. Pass C-P1..3 and C-F1..6.
8. No model family serves it until that family's runtime adds a table entry and passes C-R1..8.

**A new model family**
1. Add a runtime repo and its `family.yml` with an embodiment table (R1).
2. Map only from the declared spec (R3). Never pick a slot from an arm name. A frame conversion uses
   `base_poses` (R6). Do not rely on `qpos`. Read `tool`, `execution` and `gripper_state` if the
   model depends on them.
3. Refuse every embodiment and spec without an entry (R2).
4. Recipes use the serving functions (R5).
5. Pass C-R1..8, with C-R3 against the family's own training transforms.
6. The competition core and the benchmark forks do not change.
