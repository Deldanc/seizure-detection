"""Generate causal per-window features and a local EDF dataset manifest."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.anticipatory_pipeline import build_dataset, RAW_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", help="directory containing local EDFs and chb03-summary.txt")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--prediction-horizon-minutes", type=float, default=20.0,
                        help="seizure occurrence period after the SPH")
    parser.add_argument("--sph-minutes", type=float, default=5.0)
    parser.add_argument("--postictal-exclusion-minutes", type=float, default=30.0)
    parser.add_argument("--interictal-guard-minutes", type=float, default=0.0,
                        help="extra exclusion before the positive interval")
    args = parser.parse_args()
    kwargs = {"window_sec": args.window_seconds, "step_sec": args.step_seconds,
        "horizon_sec": args.prediction_horizon_minutes * 60,
        "sph_sec": args.sph_minutes * 60,
        "postictal_sec": args.postictal_exclusion_minutes * 60,
        "guard_sec": args.interictal_guard_minutes * 60}
    frame, manifest = build_dataset(raw_dir=Path(args.raw_dir) if args.raw_dir else RAW_DIR, **kwargs)
    print(f"Saved {len(frame)} windows ({int(frame.eligible.sum())} eligible; {int(frame.label.sum())} preictal) and manifest for {len(manifest)} recordings.")
