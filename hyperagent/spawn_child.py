"""Step 1: Spawn a child agent via claude CLI to generate a model file."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLAUDE_TIMEOUT = 90


def spawn_child(gen: int, k: int, prompt_file: Path) -> Path:
    print(f"\nStep 1.{k} — Spawning child {k} (gen {gen})")
    print(f"  Prompt : {prompt_file.relative_to(ROOT)}")

    model_dir = ROOT / "models" / f"gen_{gen}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"child_{k}.py"
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output : {model_file.relative_to(ROOT)}")

    spec = prompt_file.read_text()
    full_prompt = f"""Write a complete PyTorch CIFAR-10 training and evaluation script.

HARD RULES (never break these):
- Output ONLY a single Python code block (```python ... ```). No prose before or after.
- Print  PARAM_COUNT=<integer>  before training starts.
- Model must have under 1,000,000 parameters.
- Use only torch and torchvision. No other ML libraries.
- Data directory: {data_dir}
- Exit code 0 on success, non-zero on error.
- Total wall time budget is 55 seconds on CPU (no GPU). The process is hard-killed at 60s.
  A safe rule: if 1 epoch takes T seconds, use at most floor(45 / T) epochs (min 1).
  Measure the first epoch time and reduce epochs dynamically if needed.
  Use a small fast model — depthwise convs, small channels — to fit more epochs in budget.

PROGRESS PRINTING (required — use these exact formats):
- Use `import time` and record `epoch_start = time.time()` at the start of each epoch.
- After EVERY training epoch: run a full test-set evaluation, then print both lines:
    import json; print("EPOCH_JSON=" + json.dumps({{"epoch": e, "total": total_epochs, "loss": round(loss,4), "acc": round(train_acc,4), "epoch_sec": round(time.time()-epoch_start,2)}})); sys.stdout.flush()
    print(f"VAL_ACCURACY={{val_acc:.4f}}"); sys.stdout.flush()
- This ensures a valid VAL_ACCURACY is always on stdout even if the process is killed early.
- The LAST VAL_ACCURACY line printed is used as the final score.

ARCHITECTURE & TRAINING SPECIFICATION:
{spec}
"""

    print(f"  Calling claude (timeout={CLAUDE_TIMEOUT}s) ...")
    code = _run_claude(full_prompt)
    if not code:
        raise ValueError(f"No Python code found in claude output for gen={gen} child={k}\n{result.stdout[:300]}")

    model_file.write_text(code)
    print(f"  Written: {len(code)} chars")
    return model_file


def _run_claude(prompt: str) -> str:
    """Run claude --print and stream output to terminal while capturing it."""
    proc = subprocess.Popen(
        ["claude", "--print", prompt],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait(timeout=CLAUDE_TIMEOUT)
    return _extract_code("".join(lines))


def _extract_code(text: str) -> str:
    """Extract first ```python ... ``` block. Fall back to raw text if it looks like code."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fences — accept if it starts with a Python keyword
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "#!")):
        return stripped
    return ""
