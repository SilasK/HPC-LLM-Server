# LLM Proxy — Client Setup

## OpenCode Plugin: Session Headers

The server uses the `X-Session-ID` header for session-pinned routing (KV cache reuse). Each OpenCode subagent gets its own unique ID so it can be pinned to a different worker.

### One-time setup (per machine)

```bash
# 1. Create plugins directory if needed
mkdir -p ~/.config/opencode/plugins

# 2. Download the plugin
curl -o ~/.config/opencode/plugins/session-headers.ts \
  https://raw.githubusercontent.com/SilasK/HPC-LLM-Server/main/opencode-plugin-session-headers.ts
```

### 3. Add to `~/.config/opencode/opencode.json`

```json
{
  "plugin": ["./plugins/session-headers.ts"],
  "provider": {
    "my-hpc-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HPC LLM",
      "options": {
        "baseURL": "http://submit01:7535/v1",
        "apiKey": "<your-api-key>"
      },
      "models": {
        "Qwen3.6-27B-MTP": {
          "name": "Qwen 3.6 27B",
          "limit": { "context": 131072, "output": 8192 }
        }
      }
    }
  }
}
```

No npm install needed — the plugin is a plain `.ts` file loaded at runtime.

### What the plugin sends

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Session-ID` | `sessionUUID` | Per-subagent routing pin (each subagent gets its own worker) |
| `X-Agent-Type` | `agentName` | Subagent type for debugging/tracing |

Each subagent gets its own unique session ID from OpenCode, so no suffix is needed — uniqueness is guaranteed by the runtime.

### Porting to another machine

Copy just two files:

```bash
# From a machine that already has it set up:
scp ~/.config/opencode/plugins/session-headers.ts user@other-machine:~/.config/opencode/plugins/
scp ~/.config/opencode/opencode.json user@other-machine:~/.config/opencode/
```

Or re-run the `curl` command above.

---

# LLM Proxy — Stats & Debugging

Stats are logged to `llm_proxy_stats.jsonl` as newline-delimited JSON. Each line is one event.

## Event Types

| event | fields | meaning |
|-------|--------|---------|
| `pool_status` | active_workers, pool_capacity, pinned_workers, pending, total_sessions, pool_load | Snapshot every 30s |
| `request` | opencode_session, worker_session, cache_hit, status, worker_idle_before | LLM call routed to a worker |
| `request_error` | opencode_session, worker_session, error | LLM call failed |
| `worker_created` | worker_session, slurm_job_id, model | SLURM job submitted |
| `worker_ready` | worker_session, startup_seconds | Worker registered after boot |
| `worker_removed` | worker_session, reason | Worker left pool (idle_timeout, slurm_expiry, unreachable, slurm_failed, user_cancelled) |
| `pin_released` | worker_session, reason | Session pin released (idle_timeout) |

## Common Queries

### Pool capacity vs pending over time
```python
import json
lines = [json.loads(l) for l in open("llm_proxy_stats.jsonl")]
status = [d for d in lines if d["event"] == "pool_status"]
for d in status:
    print(f"{d['active_workers']} workers, cap={d['pool_capacity']}, "
          f"pending={d['pending']}, sessions={d['total_sessions']}, load={d['pool_load']}")
```

### Workers that crashed (unreachable)
```python
removed = [d for d in lines if d["event"] == "worker_removed"]
[print(f"{d['worker_session'][:8]} — {d['reason']}") for d in removed]
```

### Requests that errored
```python
errors = [d for d in lines if d["event"] == "request_error"]
for d in errors:
    print(f"{d['worker_session'][:8]}: {d['error']}")
```

### Request rate in last 15 min
```python
import time
now = time.time()
reqs = [d for d in lines if d["event"] == "request" and now - d["ts"] < 900]
print(f"Requests in last 15 min: {len(reqs)}")
```

## Pool Maintenance Cycle

Every 30s `_pool_maintenance()` runs:
1. Check pending SLURM jobs (clean up failed ones)
2. Health-check each worker (3 retries, 30s window → worker removed after ~90s downtime)
3. Release idle session pins (5 min)
4. Kill idle workers (10 min, except spare kept if recent activity)
5. Replace workers near SLURM walltime (5 min before expiry)
6. Log pool_status
7. Scale up: spawn worker if load ≥ 70% capacity
8. Recover: spawn worker if pool empty with pinned/pending sessions
9. Pre-scale: spawn spare if ≥ 2 requests in last 15 min
