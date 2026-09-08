"""Offline public-startup regressions for ownership and durable identity."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from tests.gateway.test_bounded_service import (
    ROOT, command, configuration, environment, invoke,
)
from tests.gateway.test_bounded_service_process import (
    free_port, start, kill_if_live, request,
)


def initialized(tmp_path):
    path, policy, home = configuration(tmp_path)
    result = invoke(tmp_path, path, home)
    assert result.returncode == 0, result.stderr
    return path, policy, home, json.loads(result.stdout)['session_id']


def snapshot(home):
    return {str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in home.rglob('*') if p.is_file()}


def assert_rejected_unchanged(tmp_path, path, home):
    before = snapshot(home)
    result = invoke(tmp_path, path, home)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'bounded_service_invalid' in result.stderr
    assert snapshot(home) == before


@pytest.mark.parametrize('check', [False, True])
def test_home_owner_excludes_other_ports_and_check(tmp_path, check):
    path, _, home = configuration(tmp_path)
    port = free_port()
    owner, log = start(tmp_path, path, home, port, 'owner')
    contender = None
    try:
        other_port = free_port()
        args = command(path, home, other_port) + (['--check'] if check else [])
        contender = subprocess.Popen(args, cwd=ROOT, env=environment(tmp_path, home),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = contender.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            pytest.fail('second process retained ownership of the same home')
        assert contender.returncode == 2, stdout + stderr
        assert request(port, 'GET', '/v1/bounded-runs/brun_probe')[0] == 404
    finally:
        if contender is not None and contender.poll() is None:
            contender.kill()
            contender.communicate(timeout=15)
        kill_if_live(owner, log)
    # OS ownership must be released after a crash, without deleting the lock inode.
    recovered = invoke(tmp_path, path, home)
    assert recovered.returncode == 0, recovered.stderr


@pytest.mark.parametrize('db_name', ['state.db', 'runs_idempotency.db'])
@pytest.mark.parametrize('damage', ['truncate', 'corrupt', 'empty-sqlite', 'replace-valid'])
def test_damaged_or_replaced_store_rejected_before_repair(tmp_path, db_name, damage):
    path, _, home, _ = initialized(tmp_path)
    db = home / db_name
    if damage == 'replace-valid':
        other = tmp_path / 'other'
        other.mkdir()
        _, _, other_home, _ = initialized(other)
        db.write_bytes((other_home / db_name).read_bytes())
    elif damage == 'empty-sqlite':
        db.unlink()
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE unrelated (id INTEGER)')
    else:
        db.write_bytes(b'' if damage == 'truncate' else b'not sqlite')
    assert_rejected_unchanged(tmp_path, path, home)


@pytest.mark.parametrize('damage', ['binding', 'session', 'messages-table', 'idempotency-table'])
def test_valid_sqlite_missing_authority_is_not_reseeded(tmp_path, damage):
    path, _, home, _ = initialized(tmp_path)
    name, sql = {
        'binding': ('state.db', 'DELETE FROM session_bindings'),
        'session': ('state.db', 'DELETE FROM sessions'),
        'messages-table': ('state.db', 'DROP TABLE messages'),
        'idempotency-table': ('runs_idempotency.db', 'DROP TABLE run_idempotency'),
    }[damage]
    with sqlite3.connect(home / name) as conn:
        conn.execute(sql)
    assert_rejected_unchanged(tmp_path, path, home)


@pytest.mark.parametrize('name', [
    'state.db', 'runs_idempotency.db', 'bounded-service.json',
    'state.db-wal', 'state.db-shm', 'state.db-journal',
    'runs_idempotency.db-wal', 'runs_idempotency.db-shm',
    'runs_idempotency.db-journal', 'state.db.quarantine.lock',
    'state.db.fts_rebuild.lock', 'logs/.__agent.lock',
])
def test_hardlinked_mutable_paths_never_write_outside_home(tmp_path, name):
    path, _, home, _ = initialized(tmp_path)
    target = home / name
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.write_bytes(b'outside sentinel')
    outside = tmp_path / 'outside'
    os.link(target, outside)
    original = outside.read_bytes()
    assert_rejected_unchanged(tmp_path, path, home)
    assert outside.read_bytes() == original
    assert os.path.samefile(target, outside)


@pytest.mark.windows_only
@pytest.mark.parametrize('location', ['home', 'ancestor', 'logs'])
def test_windows_junction_components_rejected(tmp_path, location):
    path, _, home, _ = initialized(tmp_path)
    if location == 'logs':
        target = home / 'logs'
        destination = tmp_path / 'outside-logs'
        target.rename(destination)
        alias, actual = target, destination
        invoked_home = home
    else:
        alias = tmp_path / 'alias'
        actual = home if location == 'home' else tmp_path
        invoked_home = alias if location == 'home' else alias / home.name
    result = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(alias), str(actual)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    before = snapshot(home)
    try:
        result = invoke(tmp_path, path, invoked_home)
        assert result.returncode == 2, result.stdout + result.stderr
        assert snapshot(home) == before
    finally:
        os.rmdir(alias)  # Remove the junction only, never its target.


@pytest.mark.linux_only
def test_symlink_ancestor_rejected(tmp_path):
    path, _, home, _ = initialized(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(home, target_is_directory=True)
    result = invoke(tmp_path, path, alias)
    assert result.returncode == 2, result.stdout + result.stderr
