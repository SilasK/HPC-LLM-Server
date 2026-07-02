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
