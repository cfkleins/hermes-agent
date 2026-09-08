"""Fail-closed, between-process bounded credential renewal. No autostart installer.

Run with explicit nonsecret paths. A persistent stop marker requires owner removal;
interrupted/uncertain generations require investigation, never automatic adoption.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import threading

from gateway.bounded_service import _absolute_path, _json_bytes, _read_capped, _validate_expiry, _provider_key
from gateway.bounded_service_storage import own_home, safe_components, safe_node


@dataclass(frozen=True)
class Config:
    policy: Path
    home: Path
    control: Path
    connection: Path
    source: Path
    marker: Path
    port: int = 8643

    @property
    def stop_file(self):
        return self.control / 'owner.stop'


def child_environment(home, token, key, inherited=None):
    inherited = os.environ if inherited is None else inherited
    allowed = {'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT', 'TEMP', 'TMP',
               'SYSTEMDRIVE'}
    result = {k: v for k, v in inherited.items() if k.upper() in allowed}
    result.update(HERMES_HOME=str(home), VCC_BOUNDED_HERMES_TOKEN=token,
                  VCC_BOUNDED_NOUS_KEY=key)
    return result


def _secret(value):
    if (type(value) is not str or not value or len(value) > 512
            or any(ord(c) < 33 or ord(c) > 126 for c in value)
            or ',' in value or '${' in value):
        raise ValueError('invalid_secret')
    return value


def load_token(path, *, unprotect=None):
    value = _json_bytes(_read_capped(_absolute_path(str(path)), 16384))
    if (type(value) is not dict or set(value) != {'version','base_url','case_selector','token_dpapi'}
            or type(value['version']) is not int or value['version'] != 1
            or value['base_url'] != 'http://127.0.0.1:8643'
            or value['case_selector'] != 'vector-primary-v1'
            or type(value['token_dpapi']) is not str):
        raise ValueError('invalid_connection')
    blob = base64.b64decode(value['token_dpapi'], validate=True)
    if unprotect is None:
        import win32crypt
        unprotect = lambda b: win32crypt.CryptUnprotectData(b, None, None, None, 0)[1]
    return _secret(unprotect(blob).decode('utf-8'))


def port_free(port):
    with socket.socket() as sock:
        if os.name == 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(('127.0.0.1', port))


def write_stop(path):
    safe_components(path)
    if not path.exists():
        with path.open('x', encoding='ascii') as stream:
            stream.write('owner_stop\n')
            stream.flush()
            os.fsync(stream.fileno())


def run_child(cfg, credential, token):
    """Return only after wait confirms actual exit and stdout has reached EOF."""
    command = [sys.executable, '-m', 'gateway.bounded_service', '--policy', str(cfg.policy),
               '--home', str(cfg.home), '--port', str(cfg.port),
               '--expires-at', str(credential.expires_at), '--stop-file', str(cfg.stop_file)]
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    child = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[1]),
        env=child_environment(cfg.home, token, credential.access_token),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **options)
    result = [False]
    def collect():
        # Never log or persist child output; cap individual records and drain all bytes.
        try:
            while True:
                line = child.stdout.readline(16385)
                if not line:
                    return
                result[0] = False
                if len(line) <= 16384 and line.endswith(b'\n'):
                    try:
                        receipt = _json_bytes(line)
                        result[0] = (type(receipt) is dict and set(receipt) == {'status','storage_closed'}
                            and receipt['status'] == 'generation_expired' and receipt['storage_closed'] is True)
                    except Exception:
                        pass
        finally:
            child.stdout.close()
    reader = threading.Thread(target=collect)
    reader.start()
    code = child.wait()
    reader.join()
    if type(code) is not int or child.returncode != code:
        raise ValueError('uncertain_exit')
    return code, result[0]


def supervise(cfg, *, credentials=None):
    previous = {}
    try:
        paths = [cfg.policy,cfg.home,cfg.control,cfg.connection,cfg.source,cfg.marker]
        for path in paths:
            _absolute_path(str(path))
        if (len(set(paths)) != len(paths) or cfg.home in cfg.control.parents
                or cfg.control in cfg.home.parents or cfg.port != 8643
                or cfg.control.name != 'control'
                or cfg.connection != cfg.control.parent / 'connection.json'
                or cfg.source != cfg.control.parent / 'private' / 'nous-login.dpapi'
                or cfg.home == cfg.source.parent or cfg.home in cfg.source.parents):
            raise ValueError('invalid_config')
        fresh = not cfg.control.exists()
        if fresh:
            cfg.control.mkdir(mode=0o700, parents=True, exist_ok=False)
        with own_home(cfg.control, fresh=fresh):
            binding = cfg.control / 'supervisor.json'
            expected = {'version': 2, 'credential_mode': 'isolated-dpapi',
                        'paths': [str(p) for p in paths], 'port': cfg.port}
            safe_components(binding)
            if fresh:
                with binding.open('x', encoding='utf-8') as stream:
                    json.dump(expected, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
            elif _json_bytes(_read_capped(binding, 16384)) != expected:
                raise ValueError('control_binding_mismatch')
            pending = cfg.control / 'generation.uncertain'
            safe_components(pending)
            safe_components(cfg.stop_file)
            if cfg.stop_file.exists():
                return 0
            if pending.exists() or cfg.marker.exists():
                return 2
            stop_failed = False
            def stop(_sig, _frame):
                nonlocal stop_failed
                try:
                    write_stop(cfg.stop_file)
                except Exception:
                    # A failed stop write must not abandon a running child or renew.
                    stop_failed = True
            previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            if credentials is None:
                from hermes_cli.bounded_nous_credentials import isolated_nous_credentials
                credentials = isolated_nous_credentials
            while not cfg.stop_file.exists() and not stop_failed:
                # Persist before auth/launch: crash or unknown outcome must block reentry.
                with pending.open('x', encoding='ascii') as stream:
                    stream.write('generation_in_progress\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                port_free(cfg.port)
                if cfg.stop_file.exists():
                    pending.unlink()
                    return 0
                token = load_token(cfg.connection)
                credential = credentials(source_path=cfg.source,
                    marker_path=cfg.marker, min_ttl_seconds=180)
                _provider_key(credential.access_token)
                _validate_expiry(credential.expires_at)
                if cfg.stop_file.exists():
                    pending.unlink()
                    return 0
                code, receipt = run_child(cfg, credential, token)
                if stop_failed:
                    return 2
                if cfg.stop_file.exists() and code == 0:
                    pending.unlink()
                    return 0
                if code != 75 or receipt is not True:
                    return 2
                # No auth until both actual exit and storage-close receipt are known.
                port_free(cfg.port)
                pending.unlink()
            return 0
    except Exception:
        # Do not emit exceptions, config, child output or credential objects.
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('policy','home','control','connection','source','marker'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--port', type=int, default=8643)
    return supervise(Config(**vars(parser.parse_args(argv))))


if __name__ == '__main__':
    raise SystemExit(main())
