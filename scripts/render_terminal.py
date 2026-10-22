"""Render captured terminal logs (reports/logs/*.log) as PNG screenshots.

Usage: python scripts/render_terminal.py <log> <png> "<command shown in prompt>"
The log content is the verbatim output of the command; nothing is edited.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; padding: 24px; background: #f9f9f7; }}
.win {{ display: inline-block; min-width: 760px; background: #1a1a19; border-radius: 10px;
        box-shadow: 0 6px 24px rgba(0,0,0,.18); overflow: hidden; }}
.bar {{ background: #2c2c2a; padding: 9px 14px; display: flex; gap: 7px; align-items: center; }}
.dot {{ width: 11px; height: 11px; border-radius: 50%; }}
.title {{ color: #898781; font: 12px -apple-system, Segoe UI, sans-serif; margin-left: 10px; }}
pre {{ margin: 0; padding: 16px 20px 18px; color: #f0efec;
       font: 13px/1.5 "DejaVu Sans Mono", Menlo, Consolas, monospace; white-space: pre-wrap;
       max-width: 1180px; }}
.prompt {{ color: #1baf7a; }} .cmd {{ color: #f0efec; font-weight: bold; }}
.ok {{ color: #1baf7a; }} .warn {{ color: #eda100; }}
</style></head><body><div class="win"><div class="bar">
<span class="dot" style="background:#e34948"></span><span class="dot" style="background:#eda100"></span>
<span class="dot" style="background:#1baf7a"></span><span class="title">fraud-drift-guard</span></div>
<pre>{body}</pre></div></body></html>"""


def colorize(line: str) -> str:
    esc = html.escape(line)
    if line.startswith("$ "):
        return f'<span class="prompt">$</span> <span class="cmd">{esc[2:]}</span>'
    for token in ("PASSED", "PROMOTED", " passed"):
        esc = esc.replace(token, f'<span class="ok">{token}</span>')
    return esc.replace("rejected", '<span class="warn">rejected</span>')


def render(log: Path, png: Path, command: str | None) -> None:
    lines = log.read_text().rstrip().splitlines()
    if command:
        lines = [f"$ {command}"] + lines
    page_html = TEMPLATE.format(body="\n".join(colorize(line) for line in lines))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2, viewport={"width": 1300, "height": 400})
        page.set_content(page_html)
        page.locator(".win").screenshot(path=str(png))
        browser.close()


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
