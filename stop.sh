#!/usr/bin/env bash
set -eo pipefail

SESSION="ollama-services"

echo "=== Stopping all services ==="

# Cancel any active Slurm sessions
echo "Checking for active Slurm sessions..."
for sid in $(curl -s http://submit01.ubelix.unibe.ch:7535/sessions 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for s in data:
        print(s.get('session_id', ''))
except: pass
" 2>/dev/null); do
    [ -n "$sid" ] && curl -s -X DELETE "http://submit01.ubelix.unibe.ch:7535/sessions/$sid" >/dev/null 2>&1 || true
done

# Kill via tmux
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux send-keys -t "$SESSION:0" C-c 2>/dev/null || true
    sleep 1
    tmux send-keys -t "$SESSION:1" C-c 2>/dev/null || true
    sleep 1
    tmux send-keys -t "$SESSION:2" C-c 2>/dev/null || true
    sleep 2
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "Tmux session '$SESSION' killed"
else
    echo "Tmux session not found, killing by process name..."
    pkill -f "cloudflared tunnel.*run baobab" 2>/dev/null || true
    pkill -f "opencode serve" 2>/dev/null || true
    pkill -f "server.py" 2>/dev/null || true
    sleep 2
    echo "Processes killed"
fi

echo "All services stopped."
