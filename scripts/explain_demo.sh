#!/usr/bin/env bash
# Asks a running API (with ANTHROPIC_API_KEY set) for an LLM case note.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${URL:-localhost:8000}
echo "\$ curl -s -X POST $URL/v1/explain -d @examples/suspicious.json | python3 -m json.tool"
curl -s -X POST "$URL/v1/explain" -H 'Content-Type: application/json' -d @examples/suspicious.json | python3 -m json.tool
