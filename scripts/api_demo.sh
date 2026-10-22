#!/usr/bin/env bash
# Calls a running API (make serve) with the example payloads, echoing each command.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${URL:-localhost:8000}
run() { echo "\$ $*"; eval "$@"; echo; }
run "curl -s $URL/health"
echo
for f in suspicious everyday invalid; do
  run "curl -s -X POST $URL/v1/score -H 'Content-Type: application/json' -d @examples/$f.json | python3 -m json.tool --compact"
done
