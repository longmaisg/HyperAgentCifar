"""Step 2: Run a child model, enforce constraints, write JSON log."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MAX_WALL_SEC = 175
MAX_PARAMS = 1_000_000


def evaluate_child(gen: int, k: int, prompt_file: Path) -> dict:
    print(f"\nStep 2.{k} — Evaluating child {k} (gen {gen})")

    model_file = ROOT / "models" / f"gen_{gen}" / f"child_{k}.py"
    log_dir = ROOT / "logs" / f"gen_{gen}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"child_{k}.json"

    print(f"  Running: {model_file.relative_to(ROOT)}")
    print(f"  Log    : {log_file.relative_to(ROOT)}")

    timed_out = False
    start = time.time()
    stdout_lines, stderr_lines = [], []
    try:
        proc = subprocess.Popen(
            ["uv", "run", str(model_file)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=ROOT,
        )
        # Stream stdout live; collect stderr quietly
        for line in proc.stdout:
            print(f"  | {line}", end="", flush=True)
            stdout_lines.append(line)
            if time.time() - start > MAX_WALL_SEC:
                proc.kill()
                timed_out = True
                break
        stderr_lines = proc.stderr.readlines()
        proc.wait()
    except Exception as e:
        timed_out = True
        stderr_lines.append(str(e))

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    wall = round(time.time() - start, 2)
    val_acc = _parse_float(stdout, "VAL_ACCURACY")
    params = _parse_int(stdout, "PARAM_COUNT")
    param_exceeded = params is not None and params > MAX_PARAMS

    log = {
        "generation": gen,
        "child": k,
        "prompt_file": str(prompt_file),
        "val_accuracy": 0.0 if (timed_out or param_exceeded) else (val_acc or 0.0),
        "param_count": params or 0,
        "wall_time_sec": wall,
        "timed_out": timed_out,
        "param_exceeded": bool(param_exceeded),
        "stderr_tail": stderr[-500:] if stderr else "",
    }
    log_file.write_text(json.dumps(log, indent=2))
    print(f"  acc={log['val_accuracy']:.4f}  params={log['param_count']}"
          f"  time={wall}s  timeout={timed_out}  over_params={param_exceeded}")
    return log


def _parse_float(text: str, key: str) -> float | None:
    for line in (text or "").splitlines():
        if line.startswith(f"{key}="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return None


def _parse_int(text: str, key: str) -> int | None:
    for line in (text or "").splitlines():
        if line.startswith(f"{key}="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return None
