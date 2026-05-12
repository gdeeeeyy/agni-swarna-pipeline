import argparse
from pathlib import Path

from src.train import train_kfold
from src.infer import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Training + Inference or Inference Only")

    parser.add_argument("--mode", choices=["train_infer", "infer"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--csv_path", required=False)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kfolds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_img_no", type=int, default=1)
    parser.add_argument("--max_img_no", type=int, default=20000)
    parser.add_argument("--model_path", required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.csv_path).resolve()
        if args.csv_path
        else Path(__file__).resolve().parent / "data" / "final_cleaned_image_sequence_cycles_11to20_79_80.csv"
    )

    if args.mode == "train_infer":
        best_model_path = train_kfold(
            csv_path=str(csv_path),
            img_dir=str(input_dir),
            save_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            kfolds=args.kfolds,
            seed=args.seed,
        )

        run_inference(
            img_dir=str(input_dir),
            save_dir=str(output_dir),
            model_path=str(best_model_path),
            min_img_no=args.min_img_no,
            max_img_no=args.max_img_no,
        )

    else:
        if args.model_path:
            model_path = Path(args.model_path).resolve()
        else:
            # Prefer best_model.pth (canonical best), fall back to per-fold models
            canonical = output_dir / "best_model.pth"
            if canonical.exists():
                model_path = canonical
            else:
                model_dir = output_dir / "models"
                model_files = sorted(model_dir.glob("best_model_fold*.pth"))
                if not model_files:
                    raise FileNotFoundError(
                        f"No trained model found. Expected {canonical} or fold models in {model_dir}"
                    )
                model_path = model_files[0]

        run_inference(
            img_dir=str(input_dir),
            save_dir=str(output_dir),
            model_path=str(model_path),
            min_img_no=args.min_img_no,
            max_img_no=args.max_img_no,
        )


if __name__ == "__main__":
    main()