import struct

import pytest

from vox.cpu_affinity import parse_windows_cores, choose_affinity


def record(mask, performance=0, group=0):
    processor = bytes([1, performance]) + bytes(20) + struct.pack('<H', 1)
    affinity = struct.pack('<QHHHH', mask, group, 0, 0, 0)
    return struct.pack('<II', 0, 48) + processor + affinity


def test_separate_physical_cores_preferred_over_sibling_threads():
    groups = parse_windows_cores(record(3) + record(12))
    assert groups == [[0, 1], [2, 3]]
    assert choose_affinity([0, 1, 2, 3], 2, groups) == [0, 2]
    assert choose_affinity([1, 3], 2, groups) == [1, 3]
    assert choose_affinity([0, 1], 2, groups) == [0, 1]
    assert choose_affinity([0, 1, 2, 3], 2, []) is None


def test_performance_class_and_unknown_layout_fallback():
    groups = parse_windows_cores(record(3, 0) + record(12, 8) + record(48, 8))
    assert choose_affinity(list(range(6)), 2, groups) == [2, 4]
    assert parse_windows_cores(record(3, group=1)) == []
    with pytest.raises(ValueError):
        parse_windows_cores(struct.pack('<II', 0, 0))
    with pytest.raises(ValueError):
        parse_windows_cores(record(3)[:-1])
