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
- Output ONLY valid Python code. No markdown fences, no prose outside comments.
- Print  PARAM_COUNT=<integer>  before training starts.
- Print  VAL_ACCURACY=<float>   after final test-set evaluation (0.0–1.0).
- Must finish in under 175 seconds total wall time.
- Model must have under 1,000,000 parameters.
- Use only torch and torchvision. No other ML libraries.
- Data directory: {data_dir}
- Exit code 0 on success, non-zero on error.

ARCHITECTURE & TRAINING SPECIFICATION:
{spec}
"""

    result = subprocess.run(
        ["claude", "--print", full_prompt],
        capture_output=True, text=True, timeout=CLAUDE_TIMEOUT,
    )

    code = result.stdout.strip()
    code = re.sub(r"^```python\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)

    model_file.write_text(code)
    print(f"  Written: {len(code)} chars")
    return model_file
