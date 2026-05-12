# main.py
"""
Entry point:
- train: python main.py train
- eval:  python main.py eval --model checkpoints/best_unet.pth
- infer: python main.py infer --model checkpoints/best_unet.pth --input some/folder --output out/folder
"""

import argparse
import os
from src.train import Trainer, default_cfg
from src.evaluate import evaluate
from src.inference import infer_folder

def run_train():
    cfg = default_cfg()
    trainer = Trainer(cfg)
    trainer.fit()

def run_eval(model_path):
    evaluate(model_path, "/home/tanvir/Desktop/01-08-2026/Stage3/data/val/input", "/home/tanvir/Desktop/01-08-2026/Stage3/data/val/target")

def run_infer(model_path, input_folder, output_folder):
    infer_folder(model_path, input_folder, output_folder)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("train")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", required=True)
    args = p.parse_args()

    if args.cmd == "train":
        run_train()
    elif args.cmd == "eval":
        run_eval(args.model)
    elif args.cmd == "infer":
        run_infer(args.model, args.input, args.output)
    else:
        p.print_help()
