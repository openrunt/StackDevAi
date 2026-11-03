"""Select separate physical cores without guessing SMT sibling numbering."""
import os
from pathlib import Path
import struct
import sys


def parse_windows_cores(data):
    """Parse 64-bit RelationProcessorCore records from the documented Windows API.

    Multi-group or unknown layouts fall back to the OS scheduler; never invent CPU IDs.
    """
    offset, cores = 0, []
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError('Truncated processor record')
        relationship, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError('Invalid processor record length')
        if relationship == 0:
            if size < 48:
                raise ValueError('Unsupported processor relationship layout')
            count = struct.unpack_from('<H', data, offset + 30)[0]
            if count != 1:
                return []
            mask, group = struct.unpack_from('<QH', data, offset + 32)
            if group != 0:
                return []
            processors = [i for i in range(64) if mask & (1 << i)]
            if processors:
                cores.append((data[offset + 9], processors))
        offset += size
    cores.sort(key=lambda row: (-row[0], min(row[1])))
    return [processors for _, processors in cores]


def physical_core_groups(allowed):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            return []
        function = ctypes.WinDLL('kernel32', use_last_error=True).GetLogicalProcessorInformationEx
        function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        function.restype = wintypes.BOOL
        size = wintypes.DWORD()
        function(0, None, ctypes.byref(size))
        if not 0 < size.value < 16 * 1024 * 1024:
            return []
        buffer = ctypes.create_string_buffer(size.value)
        if not function(0, buffer, ctypes.byref(size)):
            return []
        return parse_windows_cores(buffer.raw[:size.value])
    if sys.platform.startswith('linux'):
        groups = {}
        for cpu in allowed:
            topology = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu)) / 'topology'
            key = ((topology / 'physical_package_id').read_text().strip(), (topology / 'core_id').read_text().strip())
            if '-1' in key:
                return []
            groups.setdefault(key, []).append(cpu)
        return list(groups.values())
    return []


def choose_affinity(allowed, threads, groups):
    if not groups:
        return None
    allowed = list(dict.fromkeys(allowed))
    selected = []
    for group in groups:
        candidates = [cpu for cpu in allowed if cpu in group and cpu not in selected]
        if candidates:
            selected.append(candidates[0])
        if len(selected) == threads:
            return selected
    # Respect an inherited restriction even when it only contains SMT siblings.
    selected.extend(cpu for cpu in allowed if cpu not in selected)
    return selected[:threads] if selected else None
