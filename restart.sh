#!/usr/bin/env bash
set -eo pipefail

SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="ollama-services"
OPENCODE_PASSWORD="${OPENCODE_SERVER_PASSWORD:-OneToRuleThemAll}"
WORKDIR="${WORKDIR:-$(pwd)}"

if [ -f "$SERVER_DIR/.env" ]; then
    source "$SERVER_DIR/.env"
fi
LLM_API_KEY="${API_KEY:-${LLM_API_KEY:-$(openssl rand -hex 32)}}"

OPCODE_CONFIG="$HOME/.config/opencode/opencode.json"

services=()
if [ $# -eq 0 ]; then
    services=("cloudflare" "opencode" "llm")
else
    for arg in "$@"; do
        case "$arg" in
            all)    services=("cloudflare" "opencode" "llm") ;;
            cloudflare|cf) services+=("cloudflare") ;;
            opencode|oc)   services+=("opencode") ;;
            llm|proxy)     services+=("llm") ;;
            *)  echo "Unknown service: $arg. Use: cloudflare, opencode, llm, all" >&2; exit 1 ;;
        esac
    done
fi

get_window() {
    case "$1" in
        cloudflare) echo 0 ;;
        opencode)   echo 1 ;;
        llm)        echo 2 ;;
    esac
}

get_start_cmd() {
    local svc="$1"
    case "$svc" in
        cloudflare)
            echo "~/.cloudflared/bin/cloudflared tunnel --config ~/.cloudflared/config.yml run baobab"
            ;;
        opencode)
            echo "export OPENCODE_SERVER_PASSWORD='$OPENCODE_PASSWORD'; cd '$WORKDIR'; exec opencode serve --hostname 0.0.0.0 --port 4096"
            ;;
        llm)
            echo "cd '$SERVER_DIR'; export API_KEY='$LLM_API_KEY'; export SERVER_HOST='submit01.ubelix.unibe.ch'; export SERVER_PORT=7535; export LLAMA_BIN='/rs_scratch/users/sk25f059/llama.cpp/build/bin/llama-server'; export LLAMA_MODEL='/rs_scratch/users/sk25f059/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'; export DEFAULT_MEM='16G'; exec .pixi/envs/default/bin/python server.py"
            ;;
    esac
}

for svc in "${services[@]}"; do
    win=$(get_window "$svc")
    echo "=== Restarting $svc (window $win) ==="

    # Kill the old process in this window
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux send-keys -t "$SESSION:$win" C-c 2>/dev/null || true
        sleep 2
        tmux send-keys -t "$SESSION:$win" C-d 2>/dev/null || true
        sleep 1
    fi

    cmd=$(get_start_cmd "$svc")
    if [ "$svc" = "cloudflare" ]; then
        # Need to recreate window for cloudflared since it exits on C-c
        tmux new-window -t "$SESSION" -n "$svc" || true
        tmux send-keys -t "$SESSION:$win" "$cmd" Enter
    else
        tmux send-keys -t "$SESSION:$win" "$cmd" Enter
    fi

    sleep 2
    echo "$svc restarted"
done

echo "=== Done ==="
echo "  Attach: tmux attach -t $SESSION"
