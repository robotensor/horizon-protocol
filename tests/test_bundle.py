from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from protocol_testing import (
    TOO_BIG_FOR_A_DOUBLE,
    demonstration,
    expert,
    manifest,
    write_bundle,
)

from zerowam_protocol import BundleError, BundleSchemaError, bundle


def test_write_then_read_round_trips(tmp_path):
    record, arrays, _, _ = write_bundle(tmp_path / "unit")

    read_manifest, read_arrays = bundle.read(tmp_path / "unit")

    assert read_manifest == record
    assert read_manifest["bundle_version"] == bundle.BUNDLE_VERSION
    assert set(read_arrays) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(read_arrays[name], value)


def test_manifest_hashes_the_arrays(tmp_path):
    record, _, _, _ = write_bundle(tmp_path / "unit")

    assert record["files"][bundle.FRAMES_NPZ] == bundle.digest(
        tmp_path / "unit" / bundle.FRAMES_NPZ
    )


def test_read_refuses_a_tampered_npz(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.FRAMES_NPZ
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(BundleError, match="sha256"):
        bundle.read(tmp_path / "unit")

    manifest_read, _ = bundle.read(tmp_path / "unit", verify=False)
    assert manifest_read["task"] == "click_bell"


def test_read_refuses_another_bundle_version(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.DEMO_JSON
    record = json.loads(path.read_text())
    record["bundle_version"] = bundle.BUNDLE_VERSION + 1
    path.write_text(json.dumps(record))

    with pytest.raises(BundleError, match="bundle version"):
        bundle.read(tmp_path / "unit")


def test_read_refuses_a_manifest_missing_a_required_key(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.DEMO_JSON
    record = json.loads(path.read_text())
    del record["fingerprint_sha256"]
    path.write_text(json.dumps(record))

    with pytest.raises(BundleError, match="fingerprint_sha256"):
        bundle.read(tmp_path / "unit")


def test_write_refuses_a_demonstration_without_frames_or_times(tmp_path):
    arrays, _ = demonstration()

    with pytest.raises(BundleError, match="frames_"):
        bundle.write(
            tmp_path / "a",
            manifest=manifest(),
            arrays={"times": arrays["times"]},
        )
    with pytest.raises(BundleError, match="times"):
        bundle.write(
            tmp_path / "b",
            manifest=manifest(),
            arrays={"frames_head": arrays["frames_head"]},
        )


def test_write_refuses_a_manifest_missing_a_required_key(tmp_path):
    arrays, _ = demonstration()
    incomplete = manifest()
    del incomplete["task"]

    with pytest.raises(BundleError, match="task"):
        bundle.write(tmp_path / "unit", manifest=incomplete, arrays=arrays)


def test_the_allow_list_is_frames_and_times():
    """Decision Q4: a policy is given a demonstration's video and its times, and nothing else."""
    assert bundle.PUBLIC_PREFIXES == ("frames_",)
    assert bundle.PUBLIC_NAMES == ("times",)


def test_public_arrays_keeps_the_demonstrators_record_out():
    arrays, _ = demonstration()
    holding = {**arrays, **expert(), "scene_fingerprint": np.zeros(4), "states": np.zeros((6, 3))}

    public = bundle.public_arrays(holding)

    assert set(public) == {"frames_head", "frames_left_wrist", "times"}
    assert all(public[name] is arrays[name] for name in public)


@pytest.mark.parametrize(
    "name", ["qpos", "endpose", "actions", "ee_actions", "states", "demo_joints", "gripper_track"]
)
def test_write_refuses_any_array_outside_the_allow_list(tmp_path, name):
    arrays, _ = demonstration()
    arrays[name] = np.zeros((6, 16))

    with pytest.raises(BundleSchemaError, match="Q4") as refused:
        bundle.write(tmp_path / "unit", manifest=manifest(), arrays=arrays)
    assert repr(name) in str(refused.value)
    assert not (tmp_path / "unit" / bundle.FRAMES_NPZ).exists()


def test_read_refuses_a_bundle_whose_npz_holds_a_private_array(tmp_path):
    """Even with every hash in order: the rule is the allow-list, not the manifest."""
    write_bundle(tmp_path / "unit")
    unit = tmp_path / "unit"
    with np.load(unit / bundle.FRAMES_NPZ) as data:
        leaked = {name: data[name] for name in data.files}
    leaked["endpose"] = expert()["endpose"]
    np.savez_compressed(unit / bundle.FRAMES_NPZ, **leaked)
    record = json.loads((unit / bundle.DEMO_JSON).read_text())
    record["files"][bundle.FRAMES_NPZ] = bundle.digest(unit / bundle.FRAMES_NPZ)
    (unit / bundle.DEMO_JSON).write_text(json.dumps(record))

    with pytest.raises(BundleSchemaError, match="'endpose'.*Q4"):
        bundle.read(unit)


def test_a_schema_error_is_a_bundle_error_and_a_value_error():
    """A fork catches BundleSchemaError first (exit 2); any other BundleError is exit 4."""
    assert issubclass(BundleSchemaError, BundleError)
    assert issubclass(BundleSchemaError, ValueError)


def test_private_stays_out_of_the_manifest(tmp_path):
    record, _, _, _ = write_bundle(tmp_path / "unit")

    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "scene.json").exists()
    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz").exists()
    assert all(not name.startswith(bundle.PRIVATE_DIR) for name in record["files"])


# -- the schema, at write and at read (P1: #3) ---------------------------------------------------


def _edit_manifest(unit, change):
    path = unit / bundle.DEMO_JSON
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def _rewrite_npz(unit, arrays):
    """Replace the public npz and fix its hash, so only the schema can refuse it."""
    path = unit / bundle.FRAMES_NPZ
    np.savez_compressed(path, **arrays)  # numpy pickles object arrays here: bundle.write never does
    _edit_manifest(
        unit, lambda record: record["files"].update({bundle.FRAMES_NPZ: bundle.digest(path)})
    )


def _arrays(**changes):
    arrays, _ = demonstration()
    for name, value in changes.items():
        if value is None:
            del arrays[name]
        else:
            arrays[name] = value
    return arrays


def test_write_refuses_the_arrays_of_the_original_report(tmp_path):
    """float64 frames of shape (5, 4, 4) with two timestamps were written without a word."""
    arrays = {"frames_head": np.zeros((5, 4, 4)), "times": np.array([0.0, 0.1])}

    with pytest.raises(BundleSchemaError, match=r"frames_head is float64 \(5, 4, 4\)"):
        bundle.write(tmp_path / "unit", manifest=manifest(cameras=["head"]), arrays=arrays)
    assert not (tmp_path / "unit").exists()


BAD_ARRAYS = {
    "float frames": (_arrays(frames_head=np.zeros((6, 8, 10, 3))), "not uint8 RGB (T, H, W, 3)"),
    "grey frames": (_arrays(frames_head=np.zeros((6, 8, 10), np.uint8)), "not uint8 RGB"),
    "RGBA frames": (_arrays(frames_head=np.zeros((6, 8, 10, 4), np.uint8)), "not uint8 RGB"),
    "empty frames": (_arrays(frames_head=np.zeros((6, 0, 10, 3), np.uint8)), "not uint8 RGB"),
    "cameras of two lengths": (
        _arrays(frames_left_wrist=np.zeros((5, 8, 10, 3), np.uint8)),
        "not one T",
    ),
    "a times per frame too few": (_arrays(times=np.arange(5.0)), "times holds 5 values for 6"),
    "one frame": (
        {"frames_head": np.zeros((1, 8, 10, 3), np.uint8), "times": np.zeros(1)},
        "T >= 2",
    ),
    "float32 times": (_arrays(times=np.arange(6, dtype=np.float32)), "not float64 (T,)"),
    "2-D times": (_arrays(times=np.zeros((6, 1))), "not float64 (T,)"),
    "times going back": (_arrays(times=np.array([0.0, 0.1, 0.2, 0.15, 0.3, 0.4])), "decreases"),
    "a NaN time": (_arrays(times=np.array([0.0, np.nan, 0.2, 0.3, 0.4, 0.5])), "not finite"),
    "no frames": ({"times": np.arange(6.0)}, "no frames_<camera>"),
    "no times": (_arrays(times=None), "no times array"),
    "object frames": (
        _arrays(frames_head=np.empty((6, 8, 10, 3), dtype=object)),
        "would be pickled",
    ),
    "a camera with no name": (_arrays(frames_=np.zeros((6, 8, 10, 3), np.uint8)), "no camera"),
}


@pytest.mark.parametrize("case", sorted(BAD_ARRAYS))
def test_write_refuses_arrays_that_break_the_schema_and_writes_nothing(tmp_path, case):
    arrays, words = BAD_ARRAYS[case]

    with pytest.raises(BundleSchemaError) as refused:
        bundle.write(tmp_path / "unit", manifest=manifest(), arrays=arrays)
    assert words in str(refused.value)
    assert not (tmp_path / "unit").exists()


def test_times_may_repeat_and_need_not_start_at_zero(tmp_path):
    arrays = _arrays(times=np.array([3.0, 3.0, 3.1, 3.1, 3.2, 3.3]))

    bundle.write(tmp_path / "unit", manifest=manifest(), arrays=arrays)
    assert np.array_equal(bundle.read(tmp_path / "unit")[1]["times"], arrays["times"])


def _camera(**changes):
    camera = manifest()["camera"]
    for key, value in changes.items():
        if value is None:
            del camera[key]
        else:
            camera[key] = value
    return camera


HUMANGEN = {"name": "head", "w": 10, "h": 8, "source": "kling-v3", "video": "0001.mp4"}

BAD_MANIFESTS = {
    "camera={}": ({"camera": {}}, "camera has []; a expert demonstration's camera is"),
    "a camera naming no channel": (
        {"camera": _camera(name="humangen"), "cameras": ["humangen", "left_wrist"]},
        "camera.name 'humangen' names no frames_<camera>",
    ),
    "a D435 type": ({"camera": _camera(type="D435")}, "camera has"),
    "a position without orientation": (
        {"camera": _camera(pose=[0.0, -0.3, 1.2])},
        "not [x, y, z, qw, qx, qy, qz]",
    ),
    "a pose off unit norm": (
        {"camera": _camera(pose=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])},
        "unit norm",
    ),
    "no pose": ({"camera": _camera(pose=None)}, "camera has"),
    "a pose component out of range": (
        {"camera": _camera(pose=[0.0, -0.3, 1.2, 1e200, 0.0, 0.0, 0.0])},
        "unit norm",
    ),
    "a pose component no double can hold": (
        {"camera": _camera(pose=[0.0, -0.3, 1.2, TOO_BIG_FOR_A_DOUBLE, 0.0, 0.0, 0.0])},
        "not [x, y, z, qw, qx, qy, qz]",
    ),
    "a fovy no double can hold": (
        {"camera": _camera(fovy=TOO_BIG_FOR_A_DOUBLE)},
        "camera.fovy",
    ),
    "a camera name that is not a name": (
        {"camera": _camera(name=["head"])},
        "camera.name is ['head']",
    ),
    "a lens that is not the frames' size": (
        {"camera": _camera(w=640, h=480)},
        "frames_head is 8 x 10, not camera.h x camera.w 480 x 640",
    ),
    "a zero width": ({"camera": _camera(w=0)}, "camera.w is 0"),
    "a text fovy": ({"camera": _camera(fovy="37")}, "camera.fovy"),
    "a humangen camera with a fovy": (
        {"demo_source": "humangen", "camera": _camera()},
        "a humangen demonstration's camera is {name, w, h, source, video}",
    ),
    "a humangen camera without its video": (
        {"demo_source": "humangen", "camera": {**HUMANGEN, "video": ""}},
        "camera.video",
    ),
    "an unknown source": ({"demo_source": "teleop"}, "demo_source is 'teleop'"),
    "the primary camera second": (
        {"cameras": ["left_wrist", "head"]},
        "not the primary camera",
    ),
    "a camera missing from cameras": ({"cameras": ["head"]}, "cameras lists ['head']"),
    "a camera listed twice": ({"cameras": ["head", "head", "left_wrist"]}, "twice"),
    "cameras as text": ({"cameras": "head"}, "not a non-empty list"),
}


@pytest.mark.parametrize("case", sorted(BAD_MANIFESTS))
def test_write_refuses_a_camera_that_breaks_q13_or_q4(tmp_path, case):
    changes, words = BAD_MANIFESTS[case]
    arrays, _ = demonstration()

    with pytest.raises(BundleSchemaError) as refused:
        bundle.write(tmp_path / "unit", manifest=manifest(**changes), arrays=arrays)
    assert words in str(refused.value)
    assert "(Q13" in str(refused.value) or "(Q4" in str(refused.value)


@pytest.mark.parametrize("case", sorted(BAD_MANIFESTS))
def test_read_refuses_the_same_manifests(tmp_path, case):
    changes, words = BAD_MANIFESTS[case]
    write_bundle(tmp_path / "unit")
    _edit_manifest(tmp_path / "unit", lambda record: record.update(changes))

    with pytest.raises(BundleSchemaError) as refused:
        bundle.read(tmp_path / "unit")
    assert words in str(refused.value)


def test_cameras_after_the_primary_one_are_in_name_order(tmp_path):
    arrays, _ = demonstration(cameras=("head", "left_wrist", "a_right"))

    with pytest.raises(BundleSchemaError, match="not in name order"):
        bundle.write(
            tmp_path / "a",
            manifest=manifest(cameras=["head", "left_wrist", "a_right"]),
            arrays=arrays,
        )
    record = bundle.write(
        tmp_path / "b", manifest=manifest(cameras=["head", "a_right", "left_wrist"]), arrays=arrays
    )
    assert record["cameras"] == ["head", "a_right", "left_wrist"]


def test_a_humangen_bundle_records_its_video_not_a_lens(tmp_path):
    arrays, _ = demonstration(cameras=("human",))
    record = bundle.write(
        tmp_path / "unit",
        manifest=manifest(
            demo_source="humangen", camera={**HUMANGEN, "name": "human"}, cameras=["human"]
        ),
        arrays=arrays,
    )

    assert bundle.read(tmp_path / "unit")[0]["camera"] == record["camera"]


BAD_NPZ = {
    "float frames": (_arrays(frames_head=np.zeros((6, 8, 10, 3))), "not uint8 RGB"),
    "times going back": (_arrays(times=np.array([0.0, 0.1, 0.2, 0.15, 0.3, 0.4])), "decreases"),
    "a times per frame too few": (_arrays(times=np.arange(5.0)), "times holds 5 values"),
    "an object array": (
        _arrays(times=np.array([0.0, 0.1, 0.2, 0.3, 0.4, {"pickle": "me"}], dtype=object)),
        "would be pickled",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_NPZ))
def test_read_refuses_arrays_that_break_the_schema_with_every_hash_in_order(tmp_path, case):
    arrays, words = BAD_NPZ[case]
    write_bundle(tmp_path / "unit")
    _rewrite_npz(tmp_path / "unit", arrays)

    with pytest.raises(BundleSchemaError) as refused:
        bundle.read(tmp_path / "unit")
    assert words in str(refused.value)


def _integrity(unit):
    """Every way a written bundle's bytes can stop matching it, one per case."""
    npz = unit / bundle.FRAMES_NPZ
    return {
        "no demo.json": lambda: (unit / bundle.DEMO_JSON).unlink(),
        "demo.json not JSON": lambda: (unit / bundle.DEMO_JSON).write_text("{nope"),
        "demo.json not UTF-8": lambda: (unit / bundle.DEMO_JSON).write_bytes(b"\xff\xfe{}"),
        "demo.json a list": lambda: (unit / bundle.DEMO_JSON).write_text("[]"),
        "the npz missing": lambda: npz.unlink(),
        "the npz a directory": lambda: (npz.unlink(), npz.mkdir()),
        "the npz a symlink": lambda: (
            npz.rename(unit / "elsewhere.npz"),
            npz.symlink_to(unit / "elsewhere.npz"),
        ),
        "the npz truncated": lambda: _truncate_and_rehash(unit),
        "the npz not a zip": lambda: _garbage_and_rehash(unit),
        "a path through ..": lambda: _edit_manifest(
            unit, lambda r: r["files"].update({"../../etc/hostname": "0" * 64})
        ),
        "an absolute path": lambda: _edit_manifest(
            unit, lambda r: r["files"].update({"/etc/hostname": "0" * 64})
        ),
        "a file that is not the bundle's": lambda: _edit_manifest(
            unit, lambda r: r["files"].update({"notes.txt": "0" * 64})
        ),
        "a hash that is not a sha256": lambda: _edit_manifest(
            unit, lambda r: r["files"].update({bundle.FRAMES_NPZ: "abc"})
        ),
        "files not a mapping": lambda: _edit_manifest(unit, lambda r: r.update(files=["x"])),
        "no files": lambda: _edit_manifest(unit, lambda r: r.pop("files")),
    }


def _truncate_and_rehash(unit):
    npz = unit / bundle.FRAMES_NPZ
    npz.write_bytes(npz.read_bytes()[:200])
    _edit_manifest(unit, lambda r: r["files"].update({bundle.FRAMES_NPZ: bundle.digest(npz)}))


def _garbage_and_rehash(unit):
    npz = unit / bundle.FRAMES_NPZ
    npz.write_bytes(b"not an archive at all")
    _edit_manifest(unit, lambda r: r["files"].update({bundle.FRAMES_NPZ: bundle.digest(npz)}))


@pytest.mark.parametrize("case", sorted(_integrity(Path("."))))
@pytest.mark.parametrize("verify", [True, False])
def test_every_read_failure_is_a_bundle_error_and_not_a_schema_error(tmp_path, case, verify):
    """A fork maps BundleError to exit 4; an OSError or numpy's ValueError was a traceback.

    `verify=False` skips the hashes only: none of these bundles can be read either way.
    """
    unit = tmp_path / "unit"
    write_bundle(unit)
    _integrity(unit)[case]()

    with pytest.raises(BundleError) as refused:
        bundle.read(unit, verify=verify)
    assert type(refused.value) is BundleError, f"{type(refused.value).__name__}: {refused.value}"
    if "path" in case:
        assert "outside the bundle" in str(refused.value)


def test_a_hash_mismatch_is_a_bundle_error(tmp_path):
    write_bundle(tmp_path / "unit")
    (tmp_path / "unit" / bundle.FRAMES_NPZ).write_bytes(b"\0" * 64)

    with pytest.raises(BundleError, match="sha256") as refused:
        bundle.read(tmp_path / "unit")
    assert type(refused.value) is BundleError


def test_a_missing_preview_the_manifest_hashes_is_a_bundle_error(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / bundle.PREVIEW_MP4).write_bytes(b"a preview")
    arrays, _ = demonstration()
    record = bundle.write(unit, manifest=manifest(), arrays=arrays)
    assert record["files"][bundle.PREVIEW_MP4] == bundle.digest(unit / bundle.PREVIEW_MP4)
    (unit / bundle.PREVIEW_MP4).unlink()

    with pytest.raises(BundleError, match="demo.mp4"):
        bundle.read(unit)


def test_digest_of_a_missing_file_is_a_bundle_error(tmp_path):
    with pytest.raises(BundleError, match="cannot be read"):
        bundle.digest(tmp_path / "nothing")


def test_check_public_arrays_is_the_array_half_of_the_schema():
    arrays, _ = demonstration()
    bundle.check_public_arrays(arrays)

    with pytest.raises(BundleSchemaError, match="'qpos'.*Q4"):
        bundle.check_public_arrays({**arrays, "qpos": np.zeros((6, 14))})
    with pytest.raises(BundleSchemaError, match="decreases"):
        bundle.check_public_arrays({**arrays, "times": arrays["times"][::-1].copy()})
