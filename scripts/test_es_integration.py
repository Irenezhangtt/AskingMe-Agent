"""Run the ES contract test against a disposable Elasticsearch 8.17.2 container.

Usage: .venv/bin/python scripts/test_es_integration.py
Requires Docker. No model downloads or LLM credentials are needed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


IMAGE = 'docker.elastic.co/elasticsearch/elasticsearch:8.17.2'
ROOT = Path(__file__).resolve().parents[1]


def docker(*args, timeout=30):
    return subprocess.run(['docker', *args], check=True, text=True, capture_output=True, timeout=timeout).stdout.strip()


def main():
    name = 'askingme-es-test-' + uuid.uuid4().hex[:12]
    try:
        docker('info', '--format', '{{.ServerVersion}}')
        try:
            docker('image', 'inspect', IMAGE)
        except subprocess.CalledProcessError:
            print(f'Downloading {IMAGE}', flush=True)
            subprocess.run(['docker', 'pull', IMAGE], check=True, timeout=300)
        print(f'Starting disposable container {name}', flush=True)
        docker('run', '--detach', '--rm', '--name', name,
               '-p', '127.0.0.1::9200',
               '-e', 'discovery.type=single-node', '-e', 'xpack.security.enabled=false',
               '-e', 'ES_JAVA_OPTS=-Xms512m -Xmx512m', IMAGE, timeout=60)
        address = docker('port', name, '9200/tcp').splitlines()[0]
        url = 'http://' + address
        # Do not send local health checks through a configured outbound proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with opener.open(url + '/_cluster/health?wait_for_status=yellow&timeout=2s', timeout=4) as response:
                    health = json.load(response)
                if health['status'] in {'yellow', 'green'} and not health['timed_out']:
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        else:
            print(docker('logs', '--tail', '60', name), file=sys.stderr)
            raise RuntimeError('Elasticsearch did not become ready within 120 seconds')
        print(f'Elasticsearch ready at {url}; running the real ES contract test', flush=True)
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
             '-p', 'test_elasticsearch_integration.py', '-v'],
            cwd=ROOT, env={**os.environ, 'ES_TEST_URL': url}, timeout=120,
        )
        return result.returncode
    finally:
        # Remove only this invocation's unique container, even if a test fails.
        subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
        print(f'Cleaned up {name}', flush=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f'Elasticsearch verification failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(1)
