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
- Print  VAL_ACCURACY=<float>   after final test-set evaluation (0.0–1.0).
- Model must have under 1,000,000 parameters.
- Use only torch and torchvision. No other ML libraries.
- Data directory: {data_dir}
- Exit code 0 on success, non-zero on error.

PROGRESS PRINTING (required — so the user can see training is alive):
- Each training epoch: print  Epoch <n>/<total> loss=<f> acc=<f>
- Each test batch (every 10 batches): print  Test batch <n>/<total>
- Use sys.stdout.flush() after each print to ensure immediate output.

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
