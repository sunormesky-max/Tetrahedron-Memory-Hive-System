#!/usr/bin/env python3
"""
TetraMem Code Bridge
====================
Maps code structure to BCC lattice geometric memory.
Transforms AST/ Git events into TetraMem store/query calls.

Usage:
  # Index a codebase
  python code_bridge.py index /path/to/project --api http://localhost:8000

  # Watch for file changes
  python code_bridge.py watch /path/to/project --api http://localhost:8000

  # Query code memories
  python code_bridge.py query "how is authentication handled?" --api http://localhost:8000

  # Git diff → memory
  python code_bridge.py git-diff /path/to/project --api http://localhost:8000
"""

import argparse
import json
import os
import re
import sys
import time
import ast as pyast
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path

API_BASE = "http://localhost:8000"


def _api(method: str, path: str, body: Optional[dict] = None) -> Any:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# AST-based code structure extraction
# ═══════════════════════════════════════════════════════════════

def extract_python_structure(filepath: str, content: str) -> List[Dict]:
    items = []
    try:
        tree = pyast.parse(content)
    except SyntaxError:
        return items

    for node in pyast.walk(tree):
        if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef)):
            labels = ["code:function", f"code:module:{Path(filepath).stem}"]
            doc = pyast.get_docstring(node) or ""
            args = [a.arg for a in node.args.args]
            decorators = []
            for d in node.decorator_list:
                if isinstance(d, pyast.Name):
                    decorators.append(d.id)
                elif isinstance(d, pyast.Attribute):
                    decorators.append(f"{d.attr}")
            if decorators:
                labels.append(f"code:decorator:{','.join(decorators)}")
            items.append({
                "type": "function",
                "name": node.name,
                "file": filepath,
                "line": node.lineno,
                "args": args,
                "docstring": doc[:200],
                "decorators": decorators,
                "labels": labels,
                "weight": 1.5 if any(d in decorators for d in ["route", "app.route", "api_view"]) else 1.0,
            })
        elif isinstance(node, pyast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, pyast.Name):
                    bases.append(b.id)
                elif isinstance(b, pyast.Attribute):
                    bases.append(b.attr)
            items.append({
                "type": "class",
                "name": node.name,
                "file": filepath,
                "line": node.lineno,
                "bases": bases,
                "docstring": (pyast.get_docstring(node) or "")[:200],
                "labels": ["code:class", f"code:module:{Path(filepath).stem}"] +
                          [f"code:inherits:{b}" for b in bases],
                "weight": 1.5,
            })

    return items


def extract_js_ts_structure(filepath: str, content: str) -> List[Dict]:
    items = []
    ext = Path(filepath).suffix

    func_patterns = [
        (r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', "function"),
        (r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>', "arrow_function"),
        (r'class\s+(\w+)(?:\s+extends\s+(\w+))?', "class"),
    ]
    for pattern, kind in func_patterns:
        for m in re.finditer(pattern, content):
            line_start = content[:m.start()].count('\n') + 1
            labels = [f"code:{kind}"]
            if kind == "class":
                name = m.group(1)
                if m.group(2):
                    labels.append(f"code:inherits:{m.group(2)}")
                items.append({
                    "type": kind, "name": name, "file": filepath, "line": line_start,
                    "labels": labels, "weight": 1.5
                })
            else:
                name = m.group(1)
                items.append({
                    "type": kind, "name": name, "file": filepath, "line": line_start,
                    "labels": labels, "weight": 1.0
                })

    for m in re.finditer(r'(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']*)["\']', content):
        line_start = content[:m.start()].count('\n') + 1
        items.append({
            "type": "endpoint", "method": m.group(1).upper(), "path": m.group(2),
            "file": filepath, "line": line_start,
            "labels": ["code:endpoint", f"code:route:{m.group(1).upper()}:{m.group(2)}"],
            "weight": 2.0,
        })

    return items


def extract_file_structure(filepath: str) -> List[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        return extract_python_structure(filepath, content)
    elif ext in (".js", ".ts", ".jsx", ".tsx", ".mjs"):
        return extract_js_ts_structure(filepath, content)

    imports = []
    for m in re.finditer(r'(?:import|require|include|from)\s+["\']([^"\']+)["\']', content):
        imports.append(m.group(1))

    if imports:
        return [{
            "type": "file", "file": filepath,
            "imports": imports[:20],
            "labels": ["code:file", f"code:ext:{ext}"],
            "weight": 0.8
        }]
    return []


# ═══════════════════════════════════════════════════════════════
# Index a project into TetraMem
# ═══════════════════════════════════════════════════════════════

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "dist", "build", ".next", ".nuxt", "venv", ".venv", "env",
    ".tox", ".eggs", "egg-info", ".sass-cache", "coverage",
    "tetramem_data", "tetramem_data_v2", ".idea", ".vscode",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".swift", ".zig", ".lua", ".r", ".R",
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".json", ".graphql",
    ".html", ".css", ".scss", ".vue", ".svelte",
}


def should_index(filepath: str) -> bool:
    path = Path(filepath)
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return path.suffix.lower() in CODE_EXTENSIONS


def index_project(project_dir: str, project_name: str = "") -> Dict:
    project_name = project_name or Path(project_dir).name
    project_root = Path(project_dir).resolve()
    total_files = 0
    total_items = 0
    batch = []
    batch_size = 50

    log_info(f"Indexing project: {project_name} ({project_root})")

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not should_index(fpath):
                continue

            rel_path = os.path.relpath(fpath, project_root).replace("\\", "/")
            items = extract_file_structure(fpath)

            for item in items:
                item["file"] = rel_path
                item["project"] = project_name
                item["labels"] = item.get("labels", []) + [f"project:{project_name}"]

                content_parts = [f"[{item['type']}] {item.get('name', rel_path)}"]
                content_parts.append(f"File: {rel_path}")
                if "line" in item:
                    content_parts.append(f"Line: {item['line']}")
                if item.get("docstring"):
                    content_parts.append(f"Doc: {item['docstring']}")
                if item.get("args"):
                    content_parts.append(f"Args: {', '.join(item['args'])}")
                if item.get("bases"):
                    content_parts.append(f"Inherits: {', '.join(item['bases'])}")
                if item.get("imports"):
                    content_parts.append(f"Imports: {', '.join(item['imports'][:10])}")
                if item.get("method") and item.get("path"):
                    content_parts.append(f"Endpoint: {item['method']} {item['path']}")
                if item.get("decorators"):
                    content_parts.append(f"Decorators: {', '.join(item['decorators'])}")

                batch.append({
                    "content": "\n".join(content_parts),
                    "labels": item["labels"],
                    "weight": item.get("weight", 1.0),
                })
                total_items += 1

                if len(batch) >= batch_size:
                    _api("POST", "/api/v1/batch-store", {"items": batch})
                    batch = []

            total_files += 1
            if total_files % 50 == 0:
                log_info(f"  ... {total_files} files, {total_items} items indexed")

    if batch:
        _api("POST", "/api/v1/batch-store", {"items": batch})

    result = {"files_indexed": total_files, "items_stored": total_items, "project": project_name}
    log_info(f"Done: {total_files} files, {total_items} code items stored")
    return result


# ═══════════════════════════════════════════════════════════════
# Git diff → memory
# ═══════════════════════════════════════════════════════════════

def git_diff_to_memory(project_dir: str, project_name: str = "") -> Dict:
    import subprocess
    project_name = project_name or Path(project_dir).name

    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, cwd=project_dir, timeout=10
        )
        stat = result.stdout.strip()
        if not stat:
            return {"status": "no_changes"}
    except Exception as e:
        return {"error": str(e)}

    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--name-only"],
            capture_output=True, text=True, cwd=project_dir, timeout=10
        )
        changed_files = result.stdout.strip().split("\n")
        changed_files = [f for f in changed_files if f]
    except Exception:
        changed_files = []

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H %s"],
            capture_output=True, text=True, cwd=project_dir, timeout=10
        )
        last_commit = result.stdout.strip()
    except Exception:
        last_commit = "unknown"

    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=project_dir, timeout=30
        )
        diff_content = result.stdout[:5000]
    except Exception:
        diff_content = ""

    content = f"[git-diff] {project_name}\n"
    content += f"Commit: {last_commit}\n"
    content += f"Changed: {', '.join(changed_files[:20])}\n"
    content += f"Stat:\n{stat[:1000]}\n"
    if diff_content:
        content += f"Diff:\n{diff_content[:2000]}"

    labels = ["git:diff", f"project:{project_name}"]
    for f in changed_files[:5]:
        labels.append(f"git:file:{f}")

    resp = _api("POST", "/api/v1/store", {
        "content": content,
        "labels": labels,
        "weight": 1.0,
    })

    return {"stored": resp, "files_changed": len(changed_files), "commit": last_commit}


# ═══════════════════════════════════════════════════════════════
# File watcher
# ═══════════════════════════════════════════════════════════════

def watch_project(project_dir: str, project_name: str = ""):
    project_name = project_name or Path(project_dir).name

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log_error("watchdog not installed. Run: pip install watchdog")
        return

    class CodeChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.is_directory or not should_index(event.src_path):
                return
            rel = os.path.relpath(event.src_path, project_dir).replace("\\", "/")
            log_info(f"Changed: {rel}")
            items = extract_file_structure(event.src_path)
            for item in items:
                item["file"] = rel
                item["project"] = project_name
                item["labels"] = item.get("labels", []) + [
                    f"project:{project_name}", "code:change"
                ]
                content = f"[code-change] {item.get('name', rel)} ({item['type']})\nFile: {rel}"
                if item.get("line"):
                    content += f"\nLine: {item['line']}"
                _api("POST", "/api/v1/store", {
                    "content": content,
                    "labels": item["labels"],
                    "weight": 1.2,
                })
            if items:
                log_info(f"  Stored {len(items)} items from {rel}")

        def on_created(self, event):
            self.on_modified(event)

    observer = Observer()
    handler = CodeChangeHandler()
    observer.schedule(handler, project_dir, recursive=True)
    observer.start()
    log_info(f"Watching {project_dir} for code changes...")
    log_info("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ═══════════════════════════════════════════════════════════════
# Query helper
# ═══════════════════════════════════════════════════════════════

def query_code(query_text: str, k: int = 10) -> Dict:
    resp = _api("POST", "/api/v1/query", {
        "query": query_text,
        "k": k,
        "labels": ["code"]
    })
    return resp


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def log_info(msg: str):
    print(f"[code-bridge] {msg}", file=sys.stderr)


def log_error(msg: str):
    print(f"[code-bridge ERROR] {msg}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TetraMem Code Bridge")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index a project into TetraMem")
    idx.add_argument("path", help="Project directory")
    idx.add_argument("--name", default="", help="Project name (default: dir name)")
    idx.add_argument("--api", default=None, help="API URL")

    w = sub.add_parser("watch", help="Watch for file changes")
    w.add_argument("path", help="Project directory")
    w.add_argument("--name", default="", help="Project name")
    w.add_argument("--api", default=None, help="API URL")

    q = sub.add_parser("query", help="Query code memories")
    q.add_argument("text", help="Query text")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--api", default=None, help="API URL")

    gd = sub.add_parser("git-diff", help="Store git diff as memory")
    gd.add_argument("path", help="Project directory")
    gd.add_argument("--name", default="", help="Project name")
    gd.add_argument("--api", default=None, help="API URL")

    ih = sub.add_parser("install-hooks", help="Install git hooks into a project")
    ih.add_argument("path", help="Project directory")
    ih.add_argument("--api", default=None, help="API URL")

    args = parser.parse_args()
    if args.api:
        API_BASE = args.api.rstrip("/")

    if args.command == "index":
        result = index_project(args.path, args.name)
        print(json.dumps(result, indent=2))
    elif args.command == "watch":
        watch_project(args.path, args.name)
    elif args.command == "query":
        result = query_code(args.text, args.k)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "git-diff":
        result = git_diff_to_memory(args.path, args.name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "install-hooks":
        import shutil
        git_dir = os.path.join(args.path, ".git")
        if not os.path.isdir(git_dir):
            log_error("Not a git repo (no .git directory found)")
            sys.exit(1)
        hooks_dir = os.path.join(git_dir, "hooks")
        os.makedirs(hooks_dir, exist_ok=True)
        src_hook = os.path.join(os.path.dirname(__file__), "hooks", "post-commit")
        dst_hook = os.path.join(hooks_dir, "post-commit")
        if os.path.exists(src_hook):
            shutil.copy2(src_hook, dst_hook)
            os.chmod(dst_hook, 0o755)
            print(json.dumps({"installed": "post-commit", "path": dst_hook}))
        else:
            hook_content = '#!/bin/bash\nTETRAMEM_API_URL="${TETRAMEM_API_URL:-' + API_BASE + '}"\n'
            hook_content += 'COMMIT_MSG="$(git log -1 --format="%s")"\n'
            hook_content += 'CHANGED="$(git diff-tree --no-commit-id --name-only -r HEAD | tr "\\\\n" ",")"\n'
            hook_content += 'curl -sf -X POST "${TETRAMEM_API_URL}/api/v1/store" -H "Content-Type: application/json" '
            hook_content += '-d "{\\"content\\":\\"[git-commit] ${COMMIT_MSG} Files:${CHANGED}\\",\\"labels\\":[\\"git:commit\\"],\\"weight\\":1.0}" >/dev/null 2>&1 &\n'
            with open(dst_hook, "w") as f:
                f.write(hook_content)
            os.chmod(dst_hook, 0o755)
            print(json.dumps({"installed": "post-commit (generated)", "path": dst_hook}))
    else:
        parser.print_help()
