# SLURM System — Ubelix HPC

## Cluster Overview

Ubelix HPC cluster at University of Bern. The proxy submits GPU jobs to SLURM for llama workers.

---

## Available Partitions

### GPU Partitions

| Partition | GPU Type | GPUs/Node | RAM/Node | Nodes |
|-----------|----------|-----------|----------|-------|
| `gpu` | RTX3090 | 8 | 24 GB | 2 |
| `gpu` | RTX4090 | 8 | 24 GB | 10 |
| `gpu` | H100 | 8 | 80 GB | 5 |
| `gpu` | H200 | 8 | 141 GB | 1 |
| `gpu` | A100 | 6 | 80 GB | 1 |
| `gpu-invest` | RTX3090 | 8 | 24 GB | 2 |
| `gpu-invest` | RTX4090 | 8 | 24 GB | 10 |
| `gpu-invest` | H100 | 8 | 80 GB | 5 |
| `gpu-invest` | H200 | 8 | 141 GB | 1 |
| `gpu-invest` | A100 | 6 | 80 GB | 1 |
| `gpu-invest` | RTX Pro 6000 Blackwell | 16 | 24 GB | 1 |
| `teaching` | Same GPUs + 156 CPU nodes | — | — | — |

### CPU Partitions

| Partition | CPUs/Node | RAM/Node | Nodes |
|-----------|-----------|----------|-------|
| `epyc2` | 128+ | 960 GB+ | 60 |
| `bdw` | 20 | 120 GB | 65 |
| `cpu-invest` | 128+ | 960 GB+ | 82 |
| `icpu-*` | 192 | 1.4 TB | Various |

---

## QoS (Quality of Service)

### Available for `gratis` account

Only one QoS is available:

| QoS | Priority | Max Walltime | Notes |
|-----|----------|-------------|-------|
| `job_gpu_preemptable` | 0 (lowest) | 6 hours | Can be preempted; no cost; **only option for gratis account** |

Other QoS exist (`job_gpu`, `job_gpu_invest`, `job_gpu_short`) but are not available to the `gratis` account.

### Known Limitations

- **GPU quota**: Observed limit of ~4 GPU slots per user on `job_gpu_preemptable`. New jobs with `QOSMaxGRESPerUser` reason indicate the per-user GPU limit is reached.
- **Pending limit**: Pool can create up to `POOL_MAX_PENDING` (default 4) pending jobs at once, but stale pending jobs accumulate and block quota.
- **Preemption**: Jobs running under `job_gpu_preemptable` can be killed by higher-priority jobs. The pool detects unreachable workers and automatically replaces them.

---

## GPU Specifications

| GPU | VRAM | Architecture | Notes |
|-----|------|-------------|-------|
| RTX3090 | 24 GB | Ampere | Cheapest option |
| RTX4090 | 24 GB | Ada Lovelace | Current default for pool workers |
| H100 | 80 GB | Hopper | High-end, expensive |
| H200 | 141 GB | Hopper | High-end, expensive |
| A100 | 80 GB | Ampere | High-end, expensive |
| RTX Pro 6000 Blackwell | 24 GB | Blackwell | Available on 1 node |

Each GPU can run one `llama-server` instance. With `-np N` and `--slots M`, multiple concurrent requests share the same GPU.

---

## Worker Configuration (llama_worker.sh)

| Variable | Env Var | Default | Purpose |
|----------|---------|---------|---------|
| Parallel decoders | `LLAMA_NP` | 2 | Concurrent requests handled in parallel |
| KV cache slots | `LLAMA_SLOTS` | 4 | Total KV cache slots (includes queued) |
| GPU type | `DEFAULT_GPU` | `rtx4090:1` | GRES specification for sbatch |
| Walltime | `DEFAULT_TIME` | `00:20:00` | SLURM max walltime |
| Memory | `DEFAULT_MEM` | `16G` | SLURM memory request |
| QoS | `SLURM_QOS` | `job_gpu_preemptable` | SLURM QoS |
| Partition | `SLURM_PARTITION` | `gpu-invest` | SLURM partition |

### Recommended Configurations

| Setup | `DEFAULT_GPU` | `LLAMA_NP` | `LLAMA_SLOTS` | Use Case |
|-------|---------------|-----------|--------------|----------|
| 1× RTX4090 | `rtx4090:1` | 2 | 4 | Default — good balance |
| 1× RTX3090 | `rtx3090:1` | 2 | 4 | Cheaper, good for light load |
| 2× RTX3090 | `rtx3090:2` | 4 | 8 | Higher concurrency, more VRAM |

---

## SLURM Commands

```bash
# Check queue
squeue -u $USER

# Check queue with details
squeue -u $USER -o "%.8i %.10q %.12j %.2t %.10M %.6D %.20R"

# Check job details
scontrol show job <JOBID>

# Check partitions
sinfo -o "%P | %a | %D | %c | %m | %G"

# Cancel job
scancel <JOBID>

# Check QoS
sacctmgr show qos <QOS> format=Name,Priority,MaxWall,MaxJobs

# Check user associations
sacctmgr show user <USER> associations format=User,Account,Partition,QoS
```

---

## Common Issues

### QOSMaxGRESPerUser

New jobs won't start until existing ones release GPUs. Clean up stale pending jobs:

```bash
squeue -u $USER -t PD -h -o '%i' | xargs -r scancel
```

### Preempted Workers

When SLURM reclaims GPUs for higher-priority jobs, the worker becomes unreachable. The pool auto-detects this after 3 health-check failures (~90s) and spawns a replacement.

### Worker Crash

If `llama-server` crashes (OOM, segfault), the worker fails to register. The pool marks it as failed after 120s startup timeout.
