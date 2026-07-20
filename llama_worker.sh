#!/bin/bash
set -o errexit
set -o pipefail

echo "=== Worker starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Session: $SESSION_ID"
echo "Server: $SERVER_URL"

PORT=$(( ( RANDOM % 20000 ) + 10000 ))
MODEL="${LLAMA_MODEL:-/rs_scratch/users/sk25f059/models/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/rs_scratch/users/sk25f059/llama.cpp/build/bin/llama-server}"
NP="${LLAMA_NP:-2}"
SLOTS="${LLAMA_SLOTS:-4}"
NGPU="${LLAMA_NGPU:-all}"

export LD_LIBRARY_PATH="/software.9/software/GCCcore/14.2.0/lib64:/software.9/software/CUDA/12.8.0/lib64:${LD_LIBRARY_PATH:-}"

echo "Starting llama-server on 0.0.0.0:${PORT}"
echo "Model: ${MODEL}"
echo "Parallel decoders: ${NP}, Slots: ${SLOTS}, GPU layers: ${NGPU}"

GPU_FLAGS=""
if [ "$NGPU" = "0" ] || [ "$NGPU" = "none" ]; then
    # CPU mode: no GPU offloading
    GPU_FLAGS="--n-gpu-layers 0"
    echo "Running in CPU-only mode"
else
    # GPU mode: offload all layers
    GPU_FLAGS="--n-gpu-layers all --cache-type-k q4_0 --cache-type-v q4_0"
fi

${LLAMA_BIN} \
  --model "${MODEL}" \
  ${GPU_FLAGS} \
  --ctx-size 131072 \
  --flash-attn on \
  --reasoning on \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --alias "Qwen3.6-27B-MTP" \
  -np "${NP}" \
  --parallel-reserve "${SLOTS}" \
  &>/rs_scratch/users/sk25f059/llama_serve_${SESSION_ID}.log &
LLAMA_PID=$!
echo "llama-server PID: $LLAMA_PID"

for i in $(seq 1 120); do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${PORT}/v1/models 2>/dev/null || echo "000")
    if [ "$http_code" = "200" ]; then
        # Verify we actually get model data back
        model_count=$(curl -s http://127.0.0.1:${PORT}/v1/models 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('data',[]) or d.get('models',[])))" 2>/dev/null || echo "0")
        if [ "$model_count" -gt 0 ]; then
            echo "llama-server ready after ${i}s (models: $model_count)"
            break
        fi
    fi
    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "llama-server died during startup"
        wait $LLAMA_PID || true
        exit 1
    fi
    sleep 2
done

HOSTNAME=$(hostname)
echo "Registering with server: ${SERVER_URL}/register"
curl -s -X POST "${SERVER_URL}/register" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{\"session_id\": \"${SESSION_ID}\", \"hostname\": \"${HOSTNAME}\", \"port\": ${PORT}}" \
  || echo "Registration failed"

echo "Worker ready at http://${HOSTNAME}:${PORT}"
wait $LLAMA_PID
echo "=== Worker done ==="
