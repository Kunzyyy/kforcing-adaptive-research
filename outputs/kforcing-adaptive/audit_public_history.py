"""Read-only public release/history evidence; never downloads or changes weights."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
PIN_CODE = '706caa332a69d509b7fc2fa53ac7819f381a8c78'
PIN_MODEL = '16b984316bdd4585f17b11d0e61a58d236c2febc'
GH = 'https://api.github.com/repos/alibaba-damo-academy/K-Forcing'
HF = 'https://huggingface.co/api/models/zwave/K-Forcing'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def fetch(item):
    name, url = item
    observed = datetime.now(timezone.utc).isoformat()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'KForcing-public-reproduction-audit'})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            headers = {k.lower(): v for k, v in response.headers.items()
                       if k.lower() in {'date', 'etag', 'last-modified', 'link', 'x-ratelimit-remaining'}}
            result = dict(url=url, observed_utc=observed, status=response.status,
                          headers=headers, response_sha256=digest(raw), data=json.loads(raw))
    except Exception as exc:
        result = dict(url=url, observed_utc=observed, error=f'{type(exc).__name__}: {exc}')
    return name, result


def collect(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    requests = {
        'github_commits': GH+'/commits?per_page=100',
        'github_branches': GH+'/branches?per_page=100',
        'github_tags': GH+'/tags?per_page=100',
        'github_releases': GH+'/releases?per_page=100',
        'github_issues': GH+'/issues?state=all&per_page=100',
        'hf_model': HF+'?blobs=true',
        'hf_commits': HF+'/commits/main',
        'hf_discussions': HF+'/discussions',
    }
    with ThreadPoolExecutor(4) as pool:
        responses = dict(pool.map(fetch, requests.items()))
    for name, result in responses.items():
        write(out/(name+'.json'), result)
        print(name, result.get('status', result.get('error')), flush=True)
    commits = responses['hf_commits'].get('data', [])
    if isinstance(commits, list):
        revisions = [c.get('id') for c in commits if c.get('id')]
        # Bounded metadata requests only; a larger history requires explicit pagination review.
        with ThreadPoolExecutor(4) as pool:
            histories = dict(pool.map(fetch, [(r, HF+f'/tree/{r}?recursive=true') for r in revisions[:40]]))
        write(out/'hf_historical_trees.json', histories)
        print('HF historical metadata revisions:', len(histories), flush=True)
    write(out/'collection_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        collector_sha256=digest(Path(__file__).read_bytes()), pinned_code=PIN_CODE, pinned_model=PIN_MODEL,
        checkpoint_downloads=False, remote_writes=False,
        evidence_sha256={p.name: digest(p.read_bytes()) for p in sorted(out.glob('*.json'))}))


def git(repo, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={repo.resolve().as_posix()}',
                                    '-C', str(repo), *args])


class NormalizeAMP(ast.NodeTransformer):
    """Ignore ONLY the explicitly documented CUDA autocast API spelling migration."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) == 'torch.cuda.amp.autocast':
            node.func = ast.parse('torch.amp.autocast', mode='eval').body
            node.args.insert(0, ast.Constant('cuda'))
        return node


def symbols(source):
    tree = NormalizeAMP().visit(ast.parse(source))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[node.name+'.'+child.name] = digest(ast.dump(child, include_attributes=False).encode())
        elif isinstance(node, ast.FunctionDef):
            found[node.name] = digest(ast.dump(node, include_attributes=False).encode())
    return found


def history(args):
    out = args.out
    assert not (out/'git_history.json').exists()
    commits = git(args.repo, 'rev-list', '--reverse', PIN_CODE).decode().splitlines()
    tracked = ['models/transformer.py', 'models/pflm.py', 'models/autoregressive.py']
    inventories, versions, changes = [], {}, []
    previous = {}
    for rev in commits:
        paths = git(args.repo, 'ls-tree', '-r', '--name-only', rev).decode().splitlines()
        info = git(args.repo, 'show', '-s', '--format=%cI%n%s', rev).decode('utf-8').splitlines()
        inventories.append(dict(commit=rev, commit_time=info[0], title=info[1], paths=paths))
        for path in tracked:
            if path not in paths:
                continue
            raw = git(args.repo, 'show', f'{rev}:{path}')
            key = path+':'+digest(raw)
            if key not in versions:
                versions[key] = dict(path=path, sha256=digest(raw), first_seen_commit=rev,
                    symbols=symbols(raw.decode('utf-8')))
                source_out = out/'historical_sources'/rev/path
                source_out.parent.mkdir(parents=True, exist_ok=True)
                source_out.write_bytes(raw)
            if path in previous and previous[path] != key:
                before, after = versions[previous[path]]['symbols'], versions[key]['symbols']
                changes.append(dict(commit=rev, path=path, before_key=previous[path], after_key=key,
                    added=sorted(set(after)-set(before)), removed=sorted(set(before)-set(after)),
                    changed=sorted(k for k in set(before)&set(after) if before[k] != after[k])))
            previous[path] = key
    write(out/'git_history.json', dict(pinned_head=PIN_CODE,
        shallow=git(args.repo, 'rev-parse', '--is-shallow-repository').decode().strip(),
        commits=inventories, source_versions=versions, normalized_method_changes=changes,
        normalization='AST omits comments/locations; only torch.cuda.amp.autocast -> torch.amp.autocast(cuda) is normalized. Docstrings are retained.',
        limitations='Static source comparison is not proof of numerical equivalence across dependency versions.'))
    print(json.dumps(dict(ancestor_commits=len(commits), distinct_source_versions=len(versions),
                         method_change_records=changes), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'history'])
    parser.add_argument('--out', type=Path, default=ROOT/'results/public-history')
    parser.add_argument('--repo', type=Path, default=Path('work/K-Forcing'))
    args = parser.parse_args()
    globals()[args.mode](args)
