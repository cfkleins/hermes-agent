"""The synthetic SDK fixture must not let imports launch unguarded children."""
import json
import subprocess
import sys
import pytest

from tests.gateway.test_bounded_service import ROOT, configuration, environment


def test_fixture_blocks_subprocess_before_execution(tmp_path):
    _, _, home = configuration(tmp_path)
    code = '''
import subprocess, sys
try:
    subprocess.run([sys.executable, '-c', 'pass'], check=True)
except PermissionError:
    pass
else:
    raise AssertionError('fixture allowed subprocess execution')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            env=environment(tmp_path, home), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (tmp_path / 'fixture.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['event'] == 'subprocess_attempt'


@pytest.mark.windows_only
def test_fixture_only_allows_literal_native_version_builtin(tmp_path):
    _, _, home = configuration(tmp_path)
    code = '''
import subprocess
assert subprocess.check_output('ver', shell=True).strip()
try:
    subprocess.run('ver & ver', shell=True, check=True)
except PermissionError:
    pass
else:
    raise AssertionError('fixture allowed command chaining')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            env=environment(tmp_path, home), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (tmp_path / 'fixture.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['event'] == 'subprocess_attempt'
    native = (tmp_path / 'fixture.native-os.jsonl').read_text().splitlines()
    assert [json.loads(line) for line in native] == [{'event': 'native_windows_version_builtin'}]
