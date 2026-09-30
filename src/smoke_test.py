"""
Smoke test: a short training run that checks the whole pipeline works before
spending GPU time on a full run.

Runs train.py's training loop for smoke_test.n_iter iterations on the active
dataset, writes to outputs/smoke_test/<dataset>/ (so real results are never
overwritten), and checks that the loss actually goes down. config.yaml is only
read, never changed. Run from the project root:

    python -m src.smoke_test
    NERF_DATASET=poster python -m src.smoke_test
"""
import statistics
import sys

from src.train import main as train
from src.repo_util.LoadConfigurations import (
    ACTIVE_DATASET,
    SMOKE_N_ITERATIONS,
    SMOKE_VISUALIZE_EVERY,
    SMOKE_OUTPUT_DIR,
    SMOKE_MIN_LOSS_DROP,
)


def main():
    print(f"Smoke test on {ACTIVE_DATASET}: {SMOKE_N_ITERATIONS} iterations -> {SMOKE_OUTPUT_DIR}")
    history = train(
        n_iter=SMOKE_N_ITERATIONS,
        output_dir=SMOKE_OUTPUT_DIR,
        vis_every=SMOKE_VISUALIZE_EVERY,
        checkpoint_every=SMOKE_N_ITERATIONS,  # only the final checkpoint
    )

    # Compare the average loss over the first and last 10 % of the run.
    total = history["total"]
    window = max(10, len(total) // 10)
    start = statistics.mean(total[:window])
    end = statistics.mean(total[-window:])
    drop = 1.0 - end / start

    print(f"\nLoss: {start:.4f} (start) -> {end:.4f} (end), {drop:.0%} lower")
    if drop < SMOKE_MIN_LOSS_DROP:
        print(f"FAIL: expected at least {SMOKE_MIN_LOSS_DROP:.0%} lower. "
              f"Check the previews in {SMOKE_OUTPUT_DIR / 'previews'}.")
        sys.exit(1)
    print("PASS: the model is learning. Safe to start the full run.")


if __name__ == "__main__":
    main()