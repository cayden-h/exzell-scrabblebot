#!/usr/bin/env bash
# Keep the bot running: restart it whenever it exits.
# First run: BOT_TOKEN=<token from /team> scripts/run.sh
cd "$(dirname "$0")/.."
while true; do
  npx tsx src/index.ts
  code=$?
  if [ "$code" = "2" ]; then echo "fatal config error (exit 2), not restarting"; exit 2; fi
  echo "bot exited ($code), restarting in 2s"
  unset BOT_TOKEN
  sleep 2
done
