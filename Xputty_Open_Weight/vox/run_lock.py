"""OS-backed single-writer lock, released automatically if a worker exits."""
from contextlib import contextmanager
from functools import wraps
import os
import inspect
from pathlib import Path


@contextmanager
def run_lock(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it can let two processes lock different inodes.
    path = output.parent / (output.name + '.writer.lock')
    with path.open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('This model run already has an active writer; wait for it to stop') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def single_writer(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        output = signature.bind(*args, **kwargs).arguments['output']
        with run_lock(output):
            return function(*args, **kwargs)
    return wrapped
