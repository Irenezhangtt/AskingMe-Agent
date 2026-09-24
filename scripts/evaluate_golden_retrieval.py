"""Run golden retrieval ablations with a disposable, loopback-only ES container.

Pass --limit 24 for a smoke run; omit it to evaluate all 480 retrievable cases.
The 20 malformed-upload cases are evaluated by the upload integrity guard.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

from test_es_integration import IMAGE, ROOT, docker


def main():
    name = 'askingme-golden-es-' + uuid.uuid4().hex[:12]
    try:
        docker('run','--detach','--rm','--name',name,'-p','127.0.0.1::9200',
            '-e','discovery.type=single-node','-e','xpack.security.enabled=false',
            '-e','ES_JAVA_OPTS=-Xms512m -Xmx512m',IMAGE,timeout=60)
        url = 'http://' + docker('port',name,'9200/tcp').splitlines()[0]
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            try:
                with opener.open(url+'/_cluster/health?wait_for_status=yellow&timeout=2s',timeout=4) as response:
                    health=json.load(response)
                if health['status'] in {'yellow','green'} and not health['timed_out']:break
            except OSError:pass
            time.sleep(1)
        else:raise RuntimeError('Elasticsearch did not become healthy')
        print('Disposable Elasticsearch ready; starting real-model ablations',flush=True)
        result=subprocess.run([sys.executable,'-m','evaluation.golden.retrieval',*sys.argv[1:]],
            cwd=ROOT,env={**os.environ,'ELASTICSEARCH_URL':url},timeout=7200)
        return result.returncode
    finally:
        subprocess.run(['docker','rm','--force',name],capture_output=True,timeout=30)
        print('Cleaned up '+name,flush=True)


if __name__=='__main__':
    raise SystemExit(main())
