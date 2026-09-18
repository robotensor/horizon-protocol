"""Helpers shared by the tests: a demonstration, a bundle on disk, and a served policy."""

from __future__ import annotations

import errno
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zerowam_protocol import bundle
from zerowam_protocol.client import RemotePolicy

AUTHKEY_ENV = "ZEROWAM_TEST_AUTHKEY"

#: A number JSON carries and a double cannot hold: `json.loads("1" + "0" * 400)` is exactly this
#: Python int. `math.isfinite(it)` and `float(it)` raise `OverflowError`, which is no `ValueError`
#: and no `BundleError`, so every place this package range-tests a number is tested with it.
TOO_BIG_FOR_A_DOUBLE = int("1" + "0" * 400)

#: A path component longer than any file system's NAME_MAX: every lookup through it is
#: ENAMETOOLONG, as root too. pathlib's `is_symlink`, `is_dir`, `is_file` and `exists` swallow a
#: missing path only (ENOENT, ENOTDIR, EBADF, ELOOP), so this, like a directory with no search
#: permission (EACCES), reaches the caller of a probe that is not inside a module's own wrap.
TOO_LONG_A_NAME = "x" * 300


def refuse_lookup(monkeypatch, name: str) -> None:
    """Make every stat of a path ending in `name` fail with EACCES, as a directory of mode r--
    does for any user but root: its files are listed, and every lookup of one is refused."""
    real = Path.stat

    def refusing(self, *args, **kwargs):
        if self.as_posix().endswith("/" + name):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refusing)


def aloha_spec():
    """An `action_spec` shaped like RoboTwin's aloha-agilex one: two arms, world frame, waypoints.

    Typed here for the protocol's own tests only. A runtime's pinned spec is regenerated from the
    fork's spec function, never typed (Q3, C-R1); the base pose keeps sapien's float32 values.
    """
    base = [0.0, -0.64999998, 0.0, 0.70699999, 0.0, 0.0, 0.70699999]
    return {
        "action_type": "ee",
        "action_dim": 16,
        "arms": ["left", "right"],
        "layout": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        "frame": "world",
        "base_poses": {"left": list(base), "right": list(base)},
        "tool": {"point": "link6 flange", "approach_axis": "+x", "closing_axis": "y"},
        "execution": "waypoint",
        "control_hz": None,
        "gripper_command": "position",
        "gripper_state": "commanded",
        "state_channel": "endpose",
        "held": [],
        "native_action_dim": 16,
    }


def panda_spec():
    """An `action_spec` shaped like RoboCasa's PandaOmron one: one arm, base frame, setpoints."""
    return {
        "action_type": "ee",
        "action_dim": 8,
        "arms": ["right"],
        "layout": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        "frame": "robot_base",
        "tool": {"point": "robosuite grip_site", "approach_axis": "+z", "closing_axis": "x"},
        "execution": "setpoint",
        "control_hz": 20,
        "gripper_command": "position",
        "gripper_state": "measured",
        "state_channel": "endpose",
        "held": ["base", "torso"],
        "native_action_dim": 12,
    }


def pose_row(arms: int = 2, gripper: float = 0.5, seed: int = 3):
    """One state or action row in the Q3 layout: a unit quaternion and `g` for every arm."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(arms):
        quaternion = rng.standard_normal(4)
        quaternion /= np.linalg.norm(quaternion)
        blocks.append(np.concatenate([rng.uniform(-0.5, 0.5, 3), quaternion, [gripper]]))
    return np.concatenate(blocks)


def demonstration(steps: int = 6, cameras=("head", "left_wrist")):
    """A demonstration as a policy is given it (Q4): frames and times, at a native, uneven rate.

    The `info` beside them is Q14's: the demonstration's own channels in `demo_cameras`, the
    evaluated robot's observation cameras in `cameras`, and the action space in `action_spec`
    alone.
    """
    rng = np.random.default_rng(7)
    arrays = {
        f"frames_{camera}": rng.integers(0, 255, (steps, 8, 10, 3), np.uint8) for camera in cameras
    }
    arrays["times"] = np.cumsum(rng.uniform(0.05, 0.08, steps))
    return arrays, info(demo_cameras=list(cameras))


def info(**overrides):
    """What a fork sends beside a demonstration (Q14), matching `observation()`'s cameras."""
    record = {
        "embodiment": "aloha-agilex",
        "action_spec": aloha_spec(),
        "cameras": [
            {"name": "head", "role": "ego", "w": 10, "h": 8},
            {"name": "left_wrist", "role": "wrist_left", "w": 10, "h": 8},
        ],
        "demo_cameras": ["head", "left_wrist"],
        "step_limit": 400,
        "instruction": "Follow the demonstrated behavior.",
    }
    record.update(overrides)
    return record


def expert(steps: int = 6, dims: int = 14):
    """The demonstrator's record, shaped like RoboTwin's: it stays under `private/` (Q4)."""
    rng = np.random.default_rng(5)
    qpos = rng.standard_normal((steps, dims))
    endpose = np.stack([pose_row(2, seed=step) for step in range(steps)])
    return {
        "qpos": qpos,
        "endpose": endpose,
        "actions": qpos[1:].copy(),
        "ee_actions": endpose[1:].copy(),
    }


def observation(dims: int = 16, cameras=("head", "left_wrist")):
    rng = np.random.default_rng(11)
    obs = {f"frames_{camera}": rng.integers(0, 255, (8, 10, 3), np.uint8) for camera in cameras}
    obs.update(qpos=rng.standard_normal(dims), endpose=rng.standard_normal(16))
    return obs


def manifest(**overrides):
    record = {
        "unit_id": "robotwin_sim/click_bell-000",
        "axis": "robotwin_sim",
        "benchmark": "robotwin",
        "fork_commit": "0" * 40,
        "task": "click_bell",
        "category": "press_push",
        "demo_source": "expert",
        "camera": {
            "name": "head",
            "w": 10,
            "h": 8,
            "fovy": 37.0,
            "pose": [0.0, -0.3, 1.2, 0.5, -0.5, 0.5, -0.5],
        },
        "cameras": ["head", "left_wrist"],
        "action_spec": aloha_spec(),
        "task_config": "sim_clean",
        "task_config_sha256": "1" * 64,
        "scene_seed": 918273,
        "fingerprint_sha256": "2" * 64,
    }
    record.update(overrides)
    return record


def write_bundle(out_dir: Path, *, steps: int = 6, **overrides):
    """A complete bundle on disk, with the demonstrator's record under `private/`.

    `(manifest, public arrays, info, the demonstrator's record)`.
    """
    arrays, info = demonstration(steps=steps)
    record_arrays = expert(steps=steps)
    record = bundle.write(
        out_dir,
        manifest=manifest(**overrides),
        arrays=arrays,
        private={"task": "click_bell", "scene_seed": 918273},
        expert=record_arrays,
    )
    return record, arrays, info, record_arrays


def result_fields(**overrides):
    """What a fork passes `result.write` for a unit that ran to the end, as result v2 has it."""
    fields = {
        "unit_id": "robotwin_sim/click_bell-000",
        "demo_sha256": "a" * 64,
        "outcome": "success",
        "task_config": "sim_clean",
        "task_config_sha256": "1" * 64,
        "fork_commit": "0" * 40,
        "timing": {"setup_s": 1.5, "policy_s": 60.0, "sim_s": 120.0, "total_s": 182.0},
        "steps": 214,
        "step_limit": 400,
        "fingerprint_ok": True,
        "policy_calls": 7,
    }
    fields.update(overrides)
    return fields


@dataclass
class Served:
    """A policy served in its own process, and a client already connected to it."""

    process: subprocess.Popen
    policy: RemotePolicy
    log: Path
    address: str = ""
    authkey: bytes = b""
    timeout_s: float = 60.0
    client_args: dict | None = None

    def connect(self, **client_args) -> RemotePolicy:
        """Another client for the same server, as the next unit of an epoch would be.

        Sessions never overlap: close the one in hand before asking for this, or it waits in the
        listener's backlog until the server is free.
        """
        return RemotePolicy(
            self.address,
            self.authkey,
            timeout_s=self.timeout_s,
            log_file=self.log,
            **{**(self.client_args or {}), **client_args},
        )

    def close(self) -> None:
        self.policy.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def serve(
    tmp_path: Path,
    policy: str,
    *policy_args: str,
    timeout_s: float = 60.0,
    max_sessions: int = 1,
    **client_args,
) -> Served:
    """Serve `module:Class` on a Unix socket under `tmp_path` and connect to it.

    `client_args` are the client's own: its per-call timeouts (`prompt_timeout_s`,
    `act_timeout_s`) and what its `hello` declares (`action_types`, `honors_observe_every`).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    address = str(tmp_path / "policy.sock")
    log = tmp_path / "policy.log"
    # The stubs that misbehave on purpose live beside this file, so the server must see them.
    pythonpath = os.pathsep.join(
        [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env = dict(os.environ, PYTHONPATH=pythonpath, **{AUTHKEY_ENV: secrets.token_bytes(32).hex()})
    argv = [
        sys.executable,
        "-m",
        "zerowam_protocol.serve",
        "--policy",
        policy,
        "--address",
        address,
        "--authkey-env",
        AUTHKEY_ENV,
        "--log-file",
        str(log),
    ]
    for arg in policy_args:
        argv += ["--policy-arg", arg]
    argv += ["--max-sessions", str(max_sessions)]
    process = subprocess.Popen(argv, env=env)
    authkey = bytes.fromhex(env[AUTHKEY_ENV])
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if Path(address).exists():
            break
        if process.poll() is not None:
            raise AssertionError(f"the server exited with {process.returncode}")
        time.sleep(0.02)
    client = RemotePolicy(address, authkey, timeout_s=timeout_s, log_file=log, **client_args)
    served = Served(process, client, log)
    served.address = address
    served.authkey = authkey
    served.timeout_s = timeout_s
    served.client_args = dict(client_args)
    return served
