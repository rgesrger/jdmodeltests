# This doesn't fully work yet!
# There is currently no way to hit the gateway address from outside the junction instance.
#
#!/usr/bin/env bash
set -euo pipefail

# Paths (override via env vars if needed)
JUNCTION_RUN_BIN=${JUNCTION_RUN_BIN:-/users/nathanan/junction/build/junction/junction_run}
JUNCTION_CFG=${JUNCTION_CFG:-/users/nathanan/junction/build/junction/caladan_test.config}
GATEWAY_BIN=${GATEWAY_BIN:-/users/nathanan/C-and-D-final/junction-functions/build/gateway}
MODEL_PATH=${MODEL_PATH:-/users/nathanan/C-and-D-final/models/distilbert-finetuned/distilbert.onnx}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

if [[ ! -x "$JUNCTION_RUN_BIN" ]]; then
  echo "junction_run binary not found or not executable: $JUNCTION_RUN_BIN" >&2
  exit 1
fi
if [[ ! -f "$JUNCTION_CFG" ]]; then
  echo "Junction config not found: $JUNCTION_CFG" >&2
  exit 1
fi
if [[ ! -x "$GATEWAY_BIN" ]]; then
  echo "Gateway binary not found or not executable: $GATEWAY_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

set -x
"$JUNCTION_RUN_BIN" "$JUNCTION_CFG" -- \
  "$GATEWAY_BIN" \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT"
rc=$?
set +x
exit $rc
