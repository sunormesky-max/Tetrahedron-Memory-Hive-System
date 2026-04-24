import json
import os
import urllib.request
import sys
from datetime import datetime

API_URL = 'http://127.0.0.1:8000'
EXPORT_PATH = os.environ.get("TETRAMEM_EXPORT", os.path.expanduser("~/tetramem_export.md"))

def fetch_stats():
    req = urllib.request.Request(f'{API_URL}/api/v1/stats')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def fetch_all_memories(k=100):
    req = urllib.request.Request(
        f'{API_URL}/api/v1/query',
        data=json.dumps({'query': ' ', 'k': k, 'use_persistence': True}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def export_to_markdown():
    try:
        stats = fetch_stats()
        data = fetch_all_memories(k=100)
        results = data.get('results', [])

        lines = [
            '# TetraMem-XL Memory Export',
            f'',
            f'> Auto-generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'> Total memories: {stats["total_memories"]} | Labels: {stats["total_labels"]}',
            f'',
            '---',
            ''
        ]

        for r in results:
            lines.append(f'## {r["id"][:8]}')
            lines.append(f'')
            lines.append(f'{r["content"]}')
            lines.append(f'')
            labels = ', '.join(r.get('labels', []))
            if labels:
                lines.append(f'Labels: {labels}')
            lines.append(f'Score: {r.get("persistence_score", 0):.4f} | Weight: {r.get("weight", 1.0)}')
            lines.append('')
            lines.append('---')
            lines.append('')

        with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f'Exported {len(results)} memories to {EXPORT_PATH}')
    except Exception as e:
        print(f'Export failed: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    export_to_markdown()
