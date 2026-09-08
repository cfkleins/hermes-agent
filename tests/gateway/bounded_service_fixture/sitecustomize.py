"""Test interpreter hook only; the production CLI has no fixture mode."""
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
from types import SimpleNamespace as NS

if 'BOUNDED_FIXTURE_LOG' in os.environ:
    LOG = Path(os.environ['BOUNDED_FIXTURE_LOG'])
    HOME = Path(os.environ['BOUNDED_FIXTURE_HOME']).resolve()

    def record(event, **fields):
        with LOG.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, **fields)) + '\n')

    def audit(event, args):
        if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.spawn'}:
            # Python's native platform.system() needs precisely this read-only
            # Windows builtin. No PATH lookup, command chaining or other child.
            shell = os.path.join(os.environ.get('SYSTEMROOT', ''), 'System32', 'cmd.exe')
            if (os.name == 'nt' and event == 'subprocess.Popen'
                    and isinstance(args[0], str) and os.path.isabs(args[0])
                    and os.path.normcase(args[0]) == os.path.normcase(shell)
                    and args[1] == args[0] + ' /c "ver"'):
                with LOG.with_suffix('.native-os.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write('{"event":"native_windows_version_builtin"}\n')
                return
            record('subprocess_attempt', frames=[
                {'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                for frame in traceback.extract_stack(limit=12)
            ])
            raise PermissionError('fixture blocked subprocess execution')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name in {'.env', '.op.env', 'auth.json', 'active_profile', 'config.yaml',
                             'SOUL.md', 'MEMORY.md', 'USER.md', 'SKILL.md'} and not path.is_relative_to(HOME):
                record('forbidden_file', name=path.name)
                raise PermissionError('fixture blocked ambient state')
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            host = args[1][0] if event == 'socket.connect' else args[0]
            if host not in {'127.0.0.1', '::1', None}:
                record('network_attempt', host=str(host))
                raise PermissionError('fixture blocked external network')

    sys.addaudithook(audit)
    from openai.resources.chat.completions import Completions

    def create(_resource, **kwargs):
        from httpx import Timeout
        # Preserve wire fields and SDK timeout values without serializing its object.
        captured = {key: value.as_dict() if isinstance(value, Timeout) else value
                    for key, value in kwargs.items()}
        record('sdk_request', request=captured)
        message = kwargs['messages'][-1]['content']
        if message == 'fixture-sigterm':
            # Windows has no external POSIX SIGTERM delivery. Raise the real
            # registered Python signal from the SDK fixture, not a CLI endpoint.
            signal.raise_signal(signal.SIGTERM)
            time.sleep(0.5)
        reply = 'fixture answer: ' + message
        def chunk(content, finish):
            return NS(choices=[NS(index=0, delta=NS(content=content, tool_calls=None,
                      reasoning_content=None, reasoning=None), finish_reason=finish)],
                      model=kwargs['model'], usage=None)
        return iter((chunk(reply, None), chunk(None, 'stop')))

    Completions.create = create
