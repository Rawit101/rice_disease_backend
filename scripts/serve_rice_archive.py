"""Local read-only progress and dataset viewer. Binds to loopback only."""
import argparse
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('temp/rice_labeling/rice_archive_auto'))
    p.add_argument('--port',type=int,default=8770)
    args=p.parse_args(); root=args.root.resolve()
    inventory=json.loads((root/'inventory.json').read_text())
    known={r['sha256'] for r in inventory if r['source_class_id'] is not None and r['source_class_id']<9}
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):
            super().__init__(*a,directory=str(root),**kw)
        def do_GET(self):
            if self.path.split('?')[0]!='/api/progress':
                return super().do_GET()
            counts=Counter()
            for f in (root/'cache').iterdir():
                if f.suffix=='.json':
                    counts['_'.join(f.stem.split('_')[1:])]+=1
            stage={}; complete=False
            log=root.parent/'archive_job.log'
            if log.exists():
                for line in log.read_text(errors='replace').splitlines()[-100:]:
                    try:
                        item=json.loads(line)
                        if isinstance(item,dict) and 'stage' in item:
                            stage=item
                            complete=complete or item['stage']=='zip_crc_passed'
                    except ValueError:pass
            recovery=root/'recovery_selection.json'
            response=dict(total_images=len(inventory),known_unique=len(known),
                counts=dict(counts),stage=stage,finished=complete,
                comparisons_ready=(root/'manifest.json').exists(),
                recovery_unique=len(json.loads(recovery.read_text())) if recovery.exists() else None)
            data=json.dumps(response).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def log_message(self,*a):pass
    print('http://127.0.0.1:%d/progress.html'%args.port,flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
