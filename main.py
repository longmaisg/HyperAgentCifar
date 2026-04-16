"""Top-level HyperAgent loop. Usage: uv run main.py [--generations N] [--start-gen N] [--fresh]"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from hyperagent.spawn_child import spawn_child
from hyperagent.evaluate import evaluate_child
from hyperagent.evolve import evolve_generation
from hyperagent.plot import plot_generation, plot_all_generations

ROOT = Path(__file__).parent
NUM_CHILDREN = 5


def clean_all():
    """Delete all generated artifacts, keeping only the seed prompt (prompts/gen_0/)."""
    dirs = ["models", "logs", "history"]
    for d in dirs:
        path = ROOT / d
        if path.exists():
            shutil.rmtree(path)
            path.mkdir()
            (path / ".gitkeep").touch()
            print(f"  Cleared: {d}/")

    # Remove all generated prompt dirs (gen_1, gen_2, ...), keep gen_0
    prompts_dir = ROOT / "prompts"
    for p in sorted(prompts_dir.iterdir()):
        if p.is_dir() and p.name != "gen_0":
            shutil.rmtree(p)
            print(f"  Cleared: prompts/{p.name}/")

    print("  Kept   : prompts/gen_0/ (seed)")


def run_generation(gen: int):
    prompt_dir = ROOT / "prompts" / f"gen_{gen}"
    if not prompt_dir.exists():
        print(f"ERROR: No prompts for generation {gen} at {prompt_dir}")
        sys.exit(1)

    prompts = sorted(prompt_dir.glob("child_*.md"))
    if not prompts:
        print(f"ERROR: No child_*.md files in {prompt_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Generation {gen} — {len(prompts)} children to evaluate")
    print(f"Files to create: models/gen_{gen}/, logs/gen_{gen}/")
    print(f"{'='*60}")

    for prompt_file in prompts:
        k = int(prompt_file.stem.split("_")[1])
        spawn_child(gen, k, prompt_file)
        evaluate_child(gen, k, prompt_file)

    plot_generation(gen)
    plot_all_generations()

    best_acc = evolve_generation(gen)

    print(f"\nStep 4 — Git commit for generation {gen} (best_acc={best_acc:.4f})")
    subprocess.run(["git", "add", "-A"], cwd=ROOT)
    subprocess.run(
        ["git", "commit", "-m", f"gen_{gen}: best_acc={best_acc:.4f}"],
        cwd=ROOT,
    )
    subprocess.run(["git", "push"], cwd=ROOT)
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="HyperAgent CIFAR genetic loop")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--start-gen", type=int, default=0)
    parser.add_argument("--fresh", action="store_false",
                        help="Delete all models/logs/history before starting")
    args = parser.parse_args()

    if args.fresh:
        print("\nStep 0 — Fresh start: deleting all generated artifacts")
        print("  Keeping: prompts/gen_0/ (seed prompt)")
        print("  Deleting: models/, logs/, history/, prompts/gen_1+/")
        clean_all()
        print()

    for gen in range(args.start_gen, args.start_gen + args.generations):
        best = run_generation(gen)
        print(f"\n>>> Generation {gen} complete. Best val_accuracy: {best:.4f}\n")


if __name__ == "__main__":
    main()
