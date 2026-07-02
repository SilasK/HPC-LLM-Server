#!/bin/env bash
set -eo pipefail

SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCODE_PASSWORD="${OPENCODE_SERVER_PASSWORD:-OneToRuleThemAll}"
WORKDIR="${1:-$(pwd)}"
SESSION="ollama-services"

# Load or generate API key (persistent across restarts)
if [ -f "$SERVER_DIR/.env" ]; then
    source "$SERVER_DIR/.env"
fi
LLM_API_KEY="${API_KEY:-${LLM_API_KEY:-$(openssl rand -hex 32)}}"
echo "API_KEY=$LLM_API_KEY" > "$SERVER_DIR/.env"

# Sync API key into opencode provider config
OPCODE_CONFIG="$HOME/.config/opencode/opencode.json"
if [ -f "$OPCODE_CONFIG" ]; then
    sed -i 's|"apiKey": "[^"]*"|"apiKey": "'"$LLM_API_KEY"'"|' "$OPCODE_CONFIG"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

tmux new-session -d -s "$SESSION" -n services

echo ""
echo "=== LLM API Key: $LLM_API_KEY ==="
echo ""

# 1. Cloudflared tunnel
tmux send-keys -t "$SESSION:0" "~/.cloudflared/bin/cloudflared tunnel --config ~/.cloudflared/config.yml run baobab" Enter
sleep 2

# 2. OpenCode server
tmux new-window -t "$SESSION" -n opencode
tmux send-keys -t "$SESSION:1" "export OPENCODE_SERVER_PASSWORD='$OPENCODE_PASSWORD'; cd '$WORKDIR'; exec opencode serve --hostname 0.0.0.0 --port 4096" Enter
sleep 2

# 3. LLM proxy (FastAPI + Slurm)
tmux new-window -t "$SESSION" -n llm-proxy
tmux send-keys -t "$SESSION:2" \
  "cd '$SERVER_DIR'; \
   export API_KEY='$LLM_API_KEY'; \
   export SERVER_HOST='submit01.ubelix.unibe.ch'; \
   export SERVER_PORT=7535; \
   export LLAMA_BIN='/rs_scratch/users/sk25f059/llama.cpp/build/bin/llama-server'; \
    export LLAMA_MODEL='/rs_scratch/users/sk25f059/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'; \
    export DEFAULT_MEM='16G'; \
    exec .pixi/envs/default/bin/python server.py" Enter
sleep 3

echo "All services running in tmux session '$SESSION'"
echo "  Attach:  tmux attach -t $SESSION"
echo "  Stop:    ./stop.sh"
echo "  Test:    curl -H 'Authorization: Bearer $LLM_API_KEY' http://submit01:7535/v1/models"
