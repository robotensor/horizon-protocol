"""Helpers shared by the tests: a demonstration, a bundle on disk, and a served policy."""

from __future__ import annotations

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


def demonstration(steps: int = 6, dims: int = 16, cameras=("head", "left_wrist")):
    """A demonstration shaped like a RoboTwin one, at a native, uneven frame rate."""
    rng = np.random.default_rng(7)
    arrays = {
        f"frames_{camera}": rng.integers(0, 255, (steps, 8, 10, 3), np.uint8) for camera in cameras
    }
    arrays.update(
        qpos=rng.standard_normal((steps, dims)),
        endpose=rng.standard_normal((steps, 16)),
        actions=rng.standard_normal((steps - 1, dims)),
        times=np.cumsum(rng.uniform(0.05, 0.08, steps)),
    )
    info = {
        "cameras": list(cameras),
        "embodiment": "aloha-agilex",
        "action_type": "ee",
        "action_dim": 16,
        "control_hz": 250.0 / 15.0,
    }
    return arrays, info


def observation(dims: int = 16, cameras=("head", "left_wrist")):
    rng = np.random.default_rng(11)
    obs = {f"frames_{camera}": rng.integers(0, 255, (8, 10, 3), np.uint8) for camera in cameras}
    obs.update(qpos=rng.standard_normal(dims), endpose=rng.standard_normal(16))
    return obs


def manifest(**overrides):
    record = {
        "axis": "robotwin_sim",
        "benchmark": "robotwin",
        "fork_commit": "0" * 40,
        "task": "click_bell",
        "category": "press_push",
        "demo_source": "expert",
        "camera": {"name": "head_camera", "w": 10, "h": 8, "fovy": 37},
        "task_config": "sim_clean",
        "task_config_sha256": "1" * 64,
        "scene_seed": 918273,
        "fingerprint_sha256": "2" * 64,
    }
    record.update(overrides)
    return record


def write_bundle(out_dir: Path, *, steps: int = 6, **overrides):
    """A complete bundle on disk, with an expert trajectory under `private/`."""
    arrays, info = demonstration(steps=steps)
    record = bundle.write(
        out_dir,
        manifest=manifest(**overrides),
        arrays=arrays,
        private={"task": "click_bell", "scene_seed": 918273},
    )
    np.savez(Path(out_dir) / bundle.PRIVATE_DIR / "expert.npz", actions=arrays["actions"].copy())
    return record, arrays, info


@dataclass
class Served:
    """A policy served in its own process, and a client already connected to it."""

    process: subprocess.Popen
    policy: RemotePolicy
    log: Path
    address: str = ""
    authkey: bytes = b""
    timeout_s: float = 60.0

    def connect(self) -> RemotePolicy:
        """Another client for the same server, as the next unit of an epoch would be.

        Sessions never overlap: close the one in hand before asking for this, or it waits in the
        listener's backlog until the server is free.
        """
        return RemotePolicy(self.address, self.authkey, timeout_s=self.timeout_s, log_file=self.log)

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
) -> Served:
    """Serve `module:Class` on a Unix socket under `tmp_path` and connect to it."""
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
    client = RemotePolicy(address, authkey, timeout_s=timeout_s, log_file=log)
    served = Served(process, client, log)
    served.address = address
    served.authkey = authkey
    served.timeout_s = timeout_s
    return served
