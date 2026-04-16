"""Step 2: Run a child model, hard-kill at 3 min, write JSON log."""
import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MAX_WALL_SEC = 180
MAX_PARAMS = 1_000_000


def evaluate_child(gen: int, k: int, prompt_file: Path) -> dict:
    print(f"\nStep 2.{k} — Evaluating child {k} (gen {gen})")

    model_file = ROOT / "models" / f"gen_{gen}" / f"child_{k}.py"
    log_dir = ROOT / "logs" / f"gen_{gen}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"child_{k}.json"

    print(f"  Running: {model_file.relative_to(ROOT)}")
    print(f"  Timeout: {MAX_WALL_SEC}s — will hard-kill if exceeded")

    stdout_lines, stderr_lines = [], []
    start = time.time()

    proc = subprocess.Popen(
        ["uv", "run", str(model_file)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=ROOT,
    )

    # Hard-kill after MAX_WALL_SEC regardless of output
    timer = threading.Timer(MAX_WALL_SEC, _kill, [proc])
    timer.start()
    try:
        for line in proc.stdout:
            print(f"  | {line}", end="", flush=True)
            stdout_lines.append(line)
        stderr_lines = proc.stderr.readlines()
        proc.wait()
    finally:
        timer.cancel()

    wall = round(time.time() - start, 2)
    timed_out = proc.returncode in (-9, -15)  # SIGKILL / SIGTERM

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    val_acc = _parse_float(stdout, "VAL_ACCURACY") or 0.0
    params = _parse_int(stdout, "PARAM_COUNT") or 0
    param_exceeded = params > MAX_PARAMS

    if timed_out:
        print(f"  ! Hard-killed after {wall}s")

    log = {
        "generation": gen,
        "child": k,
        "prompt_file": str(prompt_file),
        "val_accuracy": 0.0 if param_exceeded else val_acc,
        "param_count": params,
        "wall_time_sec": wall,
        "timed_out": timed_out,
        "param_exceeded": param_exceeded,
        "stderr_tail": stderr[-500:] if stderr else "",
    }
    log_file.write_text(json.dumps(log, indent=2))
    print(f"  acc={log['val_accuracy']:.4f}  params={params}"
          f"  time={wall}s  killed={timed_out}  over_params={param_exceeded}")
    return log


def _kill(proc: subprocess.Popen):
    try:
        proc.kill()
    except ProcessLookupError:
        pass


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
