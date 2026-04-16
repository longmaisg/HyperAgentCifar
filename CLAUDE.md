# HyperAgent CIFAR — Genetic Algorithm on Agent Prompts

## Overview

Each **iteration** of HyperAgent works as follows:

1. **Population** — a set of prompts, each describing how to build a CIFAR PyTorch model.
2. **Evaluation** — each prompt spawns one Claude Code agent that writes and runs the model.
3. **Selection** — Claude Code reads all `(prompt, log)` pairs and picks the best performers.
4. **Mutation** — Claude Code generates up to 5 mutated/crossed variants as the next generation.
5. **Repeat** until satisfied or budget exhausted.

---

## HARD CONSTRAINTS (never violate)

- PyTorch model for **CIFAR-10** only.
- Training + evaluation must finish in **≤ 1 minute** (wall clock). Hard-killed at 60s.
- Model must have **≤ 1 000 000 parameters**.
- Use `uv` for all Python dependency management (never `pip`).
- Keep every source file **≤ 100 lines**; split into modules if needed.
- **No shell scripts** (`.sh`). All automation must be written in Python.

---

## EXPLAIN BEFORE ACTING — MANDATORY RULE

**Before executing any step**, Claude Code MUST:

1. Print the step name and number (e.g., "Step 2 — Spawning child agents").
2. Describe in plain English exactly what it is about to do and why.
3. List all files it will create or modify.
4. Wait for no confirmation (proceed automatically), but the explanation must appear in the terminal.

This rule applies to every script, every agent spawn, every file write.

---

## Directory Structure

```
HyperAgentCifar/
├── CLAUDE.md                   # this file
├── run_hyperagent.py           # top-level automation script
├── hyperagent/
│   ├── evolve.py               # selection + mutation logic (Claude Code reads logs, picks best)
│   ├── spawn_child.py          # runs one child agent given a prompt file
│   └── evaluate.py             # trains model, writes metrics to log
├── prompts/
│   └── gen_{N}/
│       └── child_{K}.md        # prompt for generation N, child K
├── models/
│   └── gen_{N}/
│       └── child_{K}.py        # model code produced by child agent
├── logs/
│   └── gen_{N}/
│       └── child_{K}.json      # {prompt, accuracy, params, wall_time, stderr}
├── history/
│   └── gen_{N}_summary.json    # per-generation winner + full leaderboard
└── data/                       # CIFAR-10 raw download cache
```

---

## Step-by-Step Protocol

### Step 0 — Bootstrap

- Create directory structure above.
- Write the **seed prompt** to `prompts/gen_0/child_0.md`.
- Install dependencies via `uv`.

### Step 1 — Spawn Children (per generation N)

For each child K (0..4):

1. Read `prompts/gen_{N}/child_{K}.md`.
2. Run `spawn_child.py N K` — calls the `claude` CLI with the prompt, which generates `models/gen_{N}/child_{K}.py`.
3. Run `evaluate.py gen_{N} child_{K}` — trains the model, enforces time/param limits, writes `logs/gen_{N}/child_{K}.json`.

### Step 2 — Evaluate & Log

`evaluate.py` must write a JSON log with:

```json
{
  "generation": 0,
  "child": 0,
  "prompt_file": "prompts/gen_0/child_0.md",
  "val_accuracy": 0.72,
  "param_count": 850000,
  "wall_time_sec": 142.3,
  "timed_out": false,
  "param_exceeded": false,
  "stderr_tail": "..."
}
```

Disqualify (accuracy = 0) if `timed_out` or `param_exceeded`.

### Step 3 — Selection & Mutation (Claude Code task)

After all children finish, `evolve.py` is called. It:

1. Reads all `logs/gen_{N}/*.json` and corresponding `prompts/gen_{N}/*.md`.
2. Ranks children by `val_accuracy` (disqualified entries ranked last).
3. Prints a ranked leaderboard to stdout and saves `history/gen_{N}_summary.json`.
4. Passes the top-K `(prompt, log)` pairs as context to Claude Code (via a structured prompt) and asks it to produce up to 5 new child prompts for generation N+1.
5. Writes new prompts to `prompts/gen_{N+1}/child_{K}.md`.

### Step 4 — Repeat

`run_hyperagent.py` loops: Step 1 → Step 2 → Step 3 → next generation.

---

## Seed Prompt Template

Every child prompt must instruct the child agent to:

1. Write a complete `models/gen_{N}/child_{K}.py` file with a PyTorch CIFAR-10 training loop.
2. Print `PARAM_COUNT=<n>` to stdout before training.
3. Print `VAL_ACCURACY=<f>` to stdout after evaluation.
4. Exit non-zero on any error.
5. Use only standard PyTorch + torchvision (no extra deps).

---

## Mutation Rules (for evolve.py prompt to Claude Code)

When asking Claude Code to generate next-generation prompts:

- **Exploit**: keep the top-1 prompt unchanged as child_0.
- **Mutate**: take top-2 prompts, change one hyperparameter or architectural choice each.
- **Crossover**: combine architectural idea from top-1 with training trick from top-2.
- **Explore**: generate 1–2 prompts that try something novel (different block type, scheduler, etc.).
- **Never** suggest prompts that violate hard constraints.

---

## Saving & History

- **Never delete** any file in `prompts/`, `models/`, `logs/`, `history/`.
- After each generation, commit everything to git via Python's `subprocess`:
  ```python
  subprocess.run(["git", "add", "-A"])
  subprocess.run(["git", "commit", "-m", f"gen_{N}: best_acc={acc}"])
  ```
- `run_hyperagent.py` handles this automatically.

---

## Running

```bash
# First time
uv run run_hyperagent.py --generations 5

# Resume from generation N
uv run run_hyperagent.py --start-gen N --generations 5
```
