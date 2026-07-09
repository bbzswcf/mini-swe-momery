#!/usr/bin/env python3

"""Render the SWE-bench-Pro issue chains as a static HTML report."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, StrictUndefined

app = typer.Typer(add_completion=False)

TEMPLATE = """\
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SWE-bench-Pro issue chains</title>
<style>
:root { color-scheme: light dark; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; background: #0e1116; color: #d8dee9; }
header { padding: 16px 24px; border-bottom: 1px solid #1f2630; background: #0b1117; position: sticky; top: 0; z-index: 10; }
header h1 { margin: 0 0 6px; font-size: 18px; }
header p { margin: 0; color: #8a96a8; }
main { display: grid; grid-template-columns: 260px 1fr; gap: 24px; padding: 24px; }
nav { position: sticky; top: 72px; align-self: start; max-height: calc(100vh - 100px); overflow: auto; }
nav h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #8a96a8; }
nav ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; }
nav a { display: block; padding: 6px 10px; border-radius: 6px; color: #c9d3e0; text-decoration: none; font-size: 12.5px; }
nav a:hover { background: #1c2330; }
nav a .meta { color: #6f7c91; font-size: 11.5px; margin-left: 6px; }
nav a.dup::after { content: " \u26A0"; color: #f0b429; }
section.chain { background: #131923; border: 1px solid #1f2630; border-radius: 10px; padding: 18px 20px; margin-bottom: 20px; }
section.chain header { padding: 0; border: none; background: none; position: static; margin-bottom: 12px; }
section.chain h2 { margin: 0; font-size: 16px; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.tag { font-size: 11.5px; padding: 2px 8px; border-radius: 999px; background: #1c2330; color: #b9c4d4; }
.tag.warn { background: #3a2a08; color: #ffcb6b; }
.tag.ok { background: #102a1a; color: #76d39a; }
.tag.info { background: #1a2540; color: #8db8ff; }
ol.timeline { list-style: none; padding: 0; margin: 14px 0 0; border-left: 2px solid #1f2630; }
ol.timeline li { position: relative; padding: 6px 0 10px 20px; }
ol.timeline li::before { content: ""; position: absolute; left: -7px; top: 12px; width: 10px; height: 10px; border-radius: 50%; background: #4c8bff; box-shadow: 0 0 0 3px #0e1116; }
ol.timeline li.dup::before { background: #f0b429; }
ol.timeline li .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }
ol.timeline li .time { font-variant-numeric: tabular-nums; color: #8a96a8; font-size: 12px; }
ol.timeline li .iid { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; color: #d8dee9; word-break: break-all; }
ol.timeline li .src { font-size: 11px; padding: 1px 6px; border-radius: 4px; background: #1c2330; color: #aab6c8; }
ol.timeline li .src.patch { background: #1a2540; color: #8db8ff; }
ol.timeline li .src.interface { background: #102a1a; color: #76d39a; }
ol.timeline li .dup-tag { font-size: 11px; padding: 1px 6px; border-radius: 4px; background: #3a2a08; color: #ffcb6b; }
ol.timeline li details { margin-top: 6px; }
ol.timeline li summary { cursor: pointer; color: #8a96a8; font-size: 12px; }
ol.timeline li .files { margin: 6px 0 0; padding: 6px 10px; background: #0b1117; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 11.5px; color: #c9d3e0; }
ol.timeline li .files li { padding: 1px 0; }
.shared { margin-top: 14px; padding: 10px 12px; background: #0b1117; border-radius: 8px; }
.shared h3 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8a96a8; }
.shared ul { margin: 0; padding-left: 18px; font-family: ui-monospace, monospace; font-size: 11.5px; }
.shared li { padding: 1px 0; }
.shared li .count { color: #f0b429; margin-left: 4px; }
</style>
</head>
<body>
<header>
  <h1>SWE-bench-Pro issue chains</h1>
  <p>
    {{ result.repo_count }} repos, {{ result.issue_count }} issues,
    {{ chains | length }} related chains (min size {{ min_chain_size }}).
    {{ dup_chain_count }} chains contain duplicate-time nodes (highlighted in yellow).
  </p>
</header>
<main>
<nav>
  <h2>Chains</h2>
  <ul>
    {% for chain in chains %}
    <li>
      <a href="#{{ chain.chain_id }}" class="{{ 'dup' if chain.has_duplicates else '' }}">
        {{ chain.chain_id }}
        <span class="meta">{{ chain.issue_count }} issues</span>
      </a>
    </li>
    {% endfor %}
  </ul>
</nav>
<div>
{% for chain in chains %}
<section class="chain" id="{{ chain.chain_id }}">
  <header>
    <h2>{{ chain.chain_id }}</h2>
    <div class="tags">
      <span class="tag info">{{ chain.repo }}</span>
      <span class="tag">{{ chain.issue_count }} issues</span>
      <span class="tag">{{ chain.start[:10] }} \u2192 {{ chain.end[:10] }}</span>
      <span class="tag ok">interface: {{ chain.source_counts.get('interface', 0) }}</span>
      <span class="tag info">patch: {{ chain.source_counts.get('patch', 0) }}</span>
      {% if chain.has_duplicates %}
      <span class="tag warn">{{ chain.duplicate_node_count }} duplicate-time nodes</span>
      {% endif %}
    </div>
  </header>
  <ol class="timeline">
    {% for issue in chain.issues %}
    <li class="{{ 'dup' if issue.is_duplicate else '' }}">
      <div class="row">
        <span class="time">{{ issue.commit_time[:19].replace('T', ' ') }}</span>
        <span class="iid">{{ issue.instance_id }}</span>
        <span class="src {{ issue.files_source }}">{{ issue.files_source }}</span>
        {% if issue.is_duplicate %}<span class="dup-tag">same time as {{ issue.duplicate_peers }}</span>{% endif %}
      </div>
      {% if issue.files %}
      <details>
        <summary>{{ issue.files | length }} files (raw: {{ issue.raw_files | length }})</summary>
        <ul class="files">
          {% for file in issue.files %}<li>{{ file }}</li>{% endfor %}
        </ul>
      </details>
      {% endif %}
    </li>
    {% endfor %}
  </ol>
  {% if chain.shared_files_top %}
  <div class="shared">
    <h3>Top shared files</h3>
    <ul>
      {% for file, count in chain.shared_files_top %}<li>{{ file }} <span class="count">\u00d7{{ count }}</span></li>{% endfor %}
    </ul>
  </div>
  {% endif %}
</section>
{% endfor %}
</div>
</main>
</body>
</html>
"""


def build_chain_view(repo: str, chain: dict) -> dict:
    time_counts = Counter(issue["commit_time"] for issue in chain["issues"])
    file_counts = Counter(file for issue in chain["issues"] for file in issue["files"])
    issues = []
    for issue in chain["issues"]:
        peers = time_counts[issue["commit_time"]]
        issues.append({**issue, "is_duplicate": peers > 1, "duplicate_peers": peers - 1})
    duplicate_node_count = sum(1 for issue in issues if issue["is_duplicate"])
    return {
        "chain_id": chain["chain_id"],
        "repo": repo,
        "issue_count": chain["issue_count"],
        "start": chain["issues"][0]["commit_time"],
        "end": chain["issues"][-1]["commit_time"],
        "issues": issues,
        "source_counts": dict(Counter(issue["files_source"] for issue in chain["issues"])),
        "has_duplicates": duplicate_node_count > 0,
        "duplicate_node_count": duplicate_node_count,
        "shared_files_top": file_counts.most_common(10),
    }


@app.command()
def main(
    input_path: Annotated[Path, typer.Option(help="Issue chains JSON to render.")] = Path(
        "data/cache/swe_bench_pro_issue_chains.json"
    ),
    output_path: Annotated[Path, typer.Option(help="Where to write the HTML report.")] = Path(
        "notes/swebench_pro_issue_chains.html"
    ),
    min_chain_size: Annotated[int, typer.Option(help="Annotate the input min chain size in the report header.")] = 4,
) -> None:
    result = json.loads(input_path.read_text())
    chains = sorted(
        (build_chain_view(repo["repo"], chain) for repo in result["repos"] for chain in repo["chains"]),
        key=lambda chain: (-chain["issue_count"], chain["repo"], chain["chain_id"]),
    )
    env = Environment(autoescape=True, undefined=StrictUndefined)
    env.filters["e"] = html.escape
    rendered = env.from_string(TEMPLATE).render(
        result=result,
        chains=chains,
        min_chain_size=min_chain_size,
        dup_chain_count=sum(1 for chain in chains if chain["has_duplicates"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(f"wrote {output_path} (chains={len(chains)}, dup_chains={sum(1 for c in chains if c['has_duplicates'])})")


if __name__ == "__main__":
    app()
