"""Step 3: Rank children, call Claude to mutate prompts for next generation."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
NUM_CHILDREN = 5
TOP_K = 3
CLAUDE_TIMEOUT = 120


def evolve_generation(gen: int) -> float:
    print(f"\nStep 3 — Reading logs for gen {gen}")
    logs = _load_logs(gen)
    logs.sort(key=lambda x: x["val_accuracy"], reverse=True)
    best_acc = logs[0]["val_accuracy"] if logs else 0.0

    _save_history(gen, logs, best_acc)
    _print_leaderboard(logs)

    top = logs[:TOP_K]
    next_prompts = _ask_claude_for_mutations(gen, top)
    _write_next_prompts(gen + 1, next_prompts)
    return best_acc


def _load_logs(gen: int) -> list[dict]:
    log_dir = ROOT / "logs" / f"gen_{gen}"
    return [json.loads(f.read_text()) for f in sorted(log_dir.glob("child_*.json"))]


def _save_history(gen: int, logs: list, best_acc: float):
    history_dir = ROOT / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generation": gen, "best_accuracy": best_acc, "leaderboard": logs}
    (history_dir / f"gen_{gen}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved : history/gen_{gen}_summary.json")


def _print_leaderboard(logs: list):
    print("  Leaderboard:")
    for i, log in enumerate(logs):
        flag = " DISQUALIFIED" if log["timed_out"] or log["param_exceeded"] else ""
        print(f"    #{i+1} child_{log['child']}: acc={log['val_accuracy']:.4f}"
              f"  params={log['param_count']}  time={log['wall_time_sec']}s{flag}")


def _ask_claude_for_mutations(gen: int, top: list[dict]) -> list[str]:
    pairs = []
    for log in top:
        spec = Path(log["prompt_file"]).read_text()
        curve = log.get("training_curve", [])
        curve_str = "\n".join(
            f"  Epoch {e['epoch']}/{e['total_epochs']} "
            f"loss={e['loss']:.4f} acc={e['acc']:.4f} "
            f"({e['epoch_sec']:.1f}s)"
            for e in curve
        ) or "  (no curve data)"
        pairs.append(
            f"### Child {log['child']} — val_acc={log['val_accuracy']:.4f}, "
            f"params={log['param_count']}, total_time={log['wall_time_sec']}s, "
            f"killed={log.get('timed_out', False)}\n"
            f"Training curve:\n{curve_str}\n\n"
            f"Prompt spec:\n{spec}"
        )

    prompt = f"""You are the mutation operator in a genetic algorithm for CIFAR-10 model prompts.

Top {len(top)} prompts from generation {gen}:

{"---".join(pairs)}

Generate exactly {NUM_CHILDREN} new child prompt specifications for generation {gen+1}.
Each prompt is natural language + structured details describing architecture and training.

Strategy:
- CHILD_0: Copy the best prompt exactly (elitism).
- CHILD_1: Mutate top-1 — change one hyperparameter (lr, batch size, scheduler, dropout).
- CHILD_2: Mutate top-2 — change one architectural choice (depth, width, block type).
- CHILD_3: Crossover — combine top-1 architecture with top-2 training strategy.
- CHILD_4: Explore — try something novel (e.g. depthwise separable convs, residual connections, cosine annealing, mixup, label smoothing).

Constraints that must never be violated:
- Under 1,000,000 parameters
- Under 175 seconds training time
- PyTorch + torchvision only

Format your response as exactly {NUM_CHILDREN} sections.
Each section MUST start with the exact marker: === CHILD_K === (where K is 0 to 4).
"""

    print("  Calling claude for mutations ...")
    proc = subprocess.Popen(
        ["claude", "--print", prompt],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines = []
    for line in proc.stdout:
        print(f"  | {line}", end="", flush=True)
        lines.append(line)
    proc.wait(timeout=CLAUDE_TIMEOUT)
    return _parse_sections("".join(lines), NUM_CHILDREN)


def _parse_sections(text: str, n: int) -> list[str]:
    sections = []
    for k in range(n):
        marker = f"=== CHILD_{k} ==="
        next_marker = f"=== CHILD_{k+1} ===" if k + 1 < n else None
        start = text.find(marker)
        if start == -1:
            sections.append(f"Fallback: simple CNN variant {k}")
            continue
        end = text.find(next_marker) if next_marker else len(text)
        sections.append(text[start + len(marker):end].strip())
    return sections


def _write_next_prompts(gen: int, prompts: list[str]):
    out_dir = ROOT / "prompts" / f"gen_{gen}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, spec in enumerate(prompts):
        path = out_dir / f"child_{k}.md"
        path.write_text(spec)
        print(f"  Wrote : prompts/gen_{gen}/child_{k}.md")
