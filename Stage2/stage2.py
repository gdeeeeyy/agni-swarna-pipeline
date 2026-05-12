import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.train import run_training
from src.infer import run_inference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train_infer", "infer"], required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mask_dir")
    p.add_argument("--csv_path")
    p.add_argument("--model_path")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--pos_weight", type=float, default=8.0)
    p.add_argument("--bce_weight", type=float, default=0.75)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--min_img_no", type=int, default=19501)
    p.add_argument("--max_img_no", type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train_infer":
        if args.mask_dir is None or args.csv_path is None:
            raise ValueError("mask_dir and csv_path are required for training")

        model_path = run_training(
            img_dir=str(img_dir),
            mask_dir=args.mask_dir,
            csv_path=args.csv_path,
            save_dir=str(output_dir),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=args.pos_weight,
            bce_weight=args.bce_weight,
            num_workers=args.num_workers,
            seed=args.seed,
            min_img_no=args.min_img_no,
            max_img_no=args.max_img_no,
        )
    else:
        if args.model_path is None:
            raise ValueError("model_path is required for inference")
        model_path = args.model_path

    run_inference(
        img_dir=str(img_dir),
        model_path=str(model_path),
        save_dir=str(output_dir),
        min_img_no=args.min_img_no,
        max_img_no=args.max_img_no,
    )


if __name__ == "__main__":
    main()