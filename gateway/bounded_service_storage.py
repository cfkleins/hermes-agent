"""Startup ownership and fail-closed storage checks for the bounded service.

The OS lock serializes cooperating bootstrap processes through drain and close.
Path checks reject pre-existing aliases; they are NOT a filesystem sandbox against
malicious same-user rename/link races. Keep the home on a local filesystem with
native locking. Never unlink the lock file (that would split its inode's owners).
"""
from contextlib import contextmanager, closing
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import tempfile
import uuid


LOCK_NAME = 'bounded-service.lock'
STORES = ('state.db', 'runs_idempotency.db')


def safe_node(path, *, missing=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError('Aliased bounded path')
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('Nonregular bounded path')
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ValueError('Hardlinked bounded path')
    return info


def safe_components(path):
    # Inspect before resolve(), which otherwise erases junction/symlink evidence.
    for component in (*reversed(path.parents), path):
        safe_node(component, missing=True)


def safe_tree(home):
    safe_components(home)
    pending = [home]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            info = safe_node(child)
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)


@contextmanager
def own_home(home, *, fresh):
    path = home / LOCK_NAME
    safe_node(path, missing=fresh)
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if fresh else 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Invalid ownership file')
        # Nonblocking acquisition is the linearization point, independent of port.
        # A byte beyond EOF can be locked on Windows without mutating the file.
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        # Closing the handle releases ownership even on startup/drain exceptions;
        # the kernel also releases it after a process crash. No PID stale cleanup.
        os.close(fd)


def _schema_digest(conn):
    rows = conn.execute(
        'SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name'
    ).fetchall()
    return sha256(repr(rows).encode('utf-8')).hexdigest()


def seal_stores(home, policy_digest, session_id):
    """Publish the manifest last; interrupted initialization stays unadoptable."""
    stores = {}
    for name in STORES:
        identity = uuid.uuid4().hex
        with closing(sqlite3.connect((home / name).as_uri() + '?mode=rw', uri=True)) as conn:
            conn.execute('CREATE TABLE bounded_service_identity '
                         '(identity TEXT PRIMARY KEY, policy_sha256 TEXT NOT NULL, '
                         'session_id TEXT NOT NULL)')
            conn.execute('INSERT INTO bounded_service_identity VALUES (?, ?, ?)',
                         (identity, policy_digest, session_id))
            conn.commit()
            stores[name] = {'identity': identity, 'schema_sha256': _schema_digest(conn)}
    return {'version': 2, 'policy_sha256': policy_digest,
            'session_id': session_id, 'stores': stores}


def validate_stores(home, marker, policy):
    """Validate isolated SQLite snapshots, never opening the originals with SQLite.

    Even mode=ro can create/change SHM sidecars. Copy the database and recovery
    inputs under the home lock, rebuild SHM in scratch space, and use mode=ro so
    no schema constructor, quarantine or reset can repair a damaged original.
    A hot rollback journal requiring writes fails closed for manual recovery.
    """
    if (type(marker) is not dict
            or set(marker) != {'version', 'policy_sha256', 'session_id', 'stores'}
            or type(marker['version']) is not int or marker['version'] != 2
            or marker['policy_sha256'] != policy.digest
            or type(marker['session_id']) is not str
            or type(marker['stores']) is not dict or set(marker['stores']) != set(STORES)):
        raise ValueError('Invalid durable manifest')
    with tempfile.TemporaryDirectory(prefix='bounded-storage-check-') as temporary:
        for name in STORES:
            source = home / name
            if not stat.S_ISREG(safe_node(source).st_mode):
                raise ValueError('Missing durable store')
            with source.open('rb') as stream:
                if stream.read(16) != b'SQLite format 3\x00':
                    raise ValueError('Invalid durable header')
            copied = Path(temporary) / name
            shutil.copyfile(source, copied)
            for suffix in ('-wal', '-journal'):
                sidecar = home / (name + suffix)
                if sidecar.exists():
                    shutil.copyfile(sidecar, Path(str(copied) + suffix))
            with closing(sqlite3.connect(copied.as_uri() + '?mode=ro', uri=True)) as conn:
                conn.execute('BEGIN')
                if (conn.execute('PRAGMA integrity_check').fetchall() != [('ok',)]
                        or conn.execute('PRAGMA foreign_key_check').fetchall()):
                    raise ValueError('Damaged durable store')
                expected = marker['stores'][name]
                if (type(expected) is not dict
                        or set(expected) != {'identity', 'schema_sha256'}
                        or _schema_digest(conn) != expected['schema_sha256']
                        or conn.execute('SELECT identity, policy_sha256, session_id '
                                        'FROM bounded_service_identity').fetchall()
                        != [(expected['identity'], policy.digest, marker['session_id'])]):
                    raise ValueError('Replaced durable store')
                if name == 'state.db':
                    values = policy.values
                    binding = conn.execute(
                        'SELECT session_id, owner_id, agent_id, case_id, source, '
                        'context_digest, version, tool_policy FROM session_bindings'
                    ).fetchall()
                    expected_binding = (marker['session_id'], values['owner_id'],
                                        values['agent_id'], values['case_id'], values['source'],
                                        values['prompt_sha256'], 1, 'deny_all')
                    root = conn.execute('SELECT id, source, user_id, parent_session_id '
                                        'FROM sessions WHERE id=?', (marker['session_id'],)).fetchall()
                    if (binding != [expected_binding]
                            or root != [(marker['session_id'], values['source'],
                                         values['owner_id'], None)]):
                        raise ValueError('Missing durable binding')
