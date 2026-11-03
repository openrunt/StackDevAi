import subprocess
import sys
from pathlib import Path

from vox.run_lock import run_lock


def test_another_process_cannot_write_active_run_and_can_enter_after_release(tmp_path):
    output = tmp_path / 'model-run'
    program = "from vox.run_lock import run_lock;import sys\ntry:\n with run_lock(sys.argv[1]): pass\nexcept RuntimeError:\n sys.exit(2)\n"
    command = [sys.executable, '-c', program, str(output)]
    root = Path(__file__).resolve().parents[1]
    with run_lock(output):
        held = subprocess.run(command, cwd=root, capture_output=True, timeout=15)
        assert held.returncode == 2, held.stderr.decode(errors='replace')
    released = subprocess.run(command, cwd=root, capture_output=True, timeout=15)
    assert released.returncode == 0, released.stderr.decode(errors='replace')
