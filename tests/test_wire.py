from __future__ import annotations

import json
from multiprocessing.connection import Pipe

import numpy as np
import pytest

from zerowam_protocol import WireError, wire


def round_trip(op, fields=None, arrays=None, **recv_kwargs):
    left, right = Pipe()
    try:
        wire.send(left, op, fields, arrays)
        return wire.recv(right, **recv_kwargs)
    finally:
        left.close()
        right.close()


def test_named_arrays_and_fields_survive():
    arrays = {
        "frames_head": np.arange(24, dtype=np.uint8).reshape(2, 2, 3, 2),
        "times": np.array([0.0, 0.07]),
        "flag": np.array([True, False]),
    }

    op, fields, received = round_trip("prompt", {"info": {"cameras": ["head"]}}, arrays)

    assert op == "prompt"
    assert fields == {"info": {"cameras": ["head"]}}
    assert set(received) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(received[name], value)
        assert received[name].dtype == value.dtype


def test_an_empty_array_survives():
    _, _, received = round_trip("act", {}, {"action": np.zeros((0, 16))})

    assert received["action"].shape == (0, 16)


def test_object_arrays_are_refused_before_anything_is_sent():
    with pytest.raises(WireError, match="dtype"):
        wire.encode("prompt", {}, {"evil": np.array([{"pickle": "me"}], dtype=object)})


def test_strings_are_refused():
    with pytest.raises(WireError, match="dtype"):
        wire.encode("act", {}, {"text": np.array(["a", "b"])})


def test_big_endian_arrives_little_endian_with_its_values_intact():
    values = np.arange(4, dtype=">i4")

    _, _, received = round_trip("act", {}, {"swapped": values})

    assert received["swapped"].dtype.byteorder in ("<", "=")
    assert np.array_equal(received["swapped"], values)


def test_fields_that_are_not_json_are_refused():
    with pytest.raises(WireError, match="plain JSON"):
        wire.encode("reset", {"seed": {1, 2}})
    with pytest.raises(WireError, match="plain JSON"):
        wire.encode("reset", {"seed": float("nan")})


def test_unknown_ops_are_refused():
    with pytest.raises(WireError, match="unknown op"):
        wire.encode("execute", {})


def test_a_header_from_another_protocol_is_refused():
    left, right = Pipe()
    try:
        header = {"protocol": wire.PROTOCOL_VERSION + 1, "op": "act", "fields": {}, "arrays": []}
        left.send_bytes(json.dumps(header).encode())
        with pytest.raises(WireError, match="protocol"):
            wire.recv(right)
    finally:
        left.close()
        right.close()


def test_a_message_over_the_receivers_limit_is_refused_before_it_is_read():
    with pytest.raises(WireError, match="bytes"):
        round_trip("act", {}, {"big": np.zeros(4096, dtype=np.float64)}, max_bytes=1024)


def test_arrays_arrive_read_only():
    _, _, received = round_trip("act", {}, {"action": np.zeros(4)})

    with pytest.raises(ValueError):
        received["action"][0] = 1.0


@pytest.mark.parametrize(
    ("address", "family"),
    [
        ("127.0.0.1:7100", "AF_INET"),
        ("/tmp/policy.sock", "AF_UNIX"),
        ("relative.sock", "AF_UNIX"),
    ],
)
def test_addresses_are_read_as_host_port_or_a_path(address, family):
    assert wire.parse_address(address)[0] == family


def test_an_empty_address_is_refused():
    with pytest.raises(WireError):
        wire.parse_address("")
