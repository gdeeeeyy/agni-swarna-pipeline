# stage3.py
"""
Entry point:
- train: python stage3.py train --csv path/to/labels.csv
- eval:  python stage3.py eval --model checkpoints/best_unet.pth --input data/val/input --target data/val/target
- infer: python stage3.py infer --model checkpoints/best_unet.pth --input some/folder --output out/folder
"""

import argparse
import os
import sys

# Ensure the parent directory is in sys.path so we can import src as a package if needed
# or just add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import Trainer, default_cfg
from src.evaluate import evaluate
from src.inference import infer_folder

def run_train(csv_path=None, train_input=None, train_target=None, val_input=None, val_target=None, checkpoint_dir=None):
    cfg = default_cfg()
    if csv_path: cfg['csv_path'] = csv_path
    if train_input: cfg['train_input'] = train_input
    if train_target: cfg['train_target'] = train_target
    if val_input: cfg['val_input'] = val_input
    if val_target: cfg['val_target'] = val_target
    if checkpoint_dir: cfg['checkpoint_dir'] = checkpoint_dir
    
    trainer = Trainer(cfg)
    trainer.fit()

def run_eval(model_path, input_dir, target_dir):
    evaluate(model_path, input_dir, target_dir)

def run_infer(model_path, input_folder, output_folder):
    infer_folder(model_path, input_folder, output_folder)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    
    t = sub.add_parser("train")
    t.add_argument("--csv", help="Path to Stage 1 CSV (optional, to filter usable images)")
    t.add_argument("--train_input", help="Directory for training input images")
    t.add_argument("--train_target", help="Directory for training target masks")
    t.add_argument("--val_input", help="Directory for validation input images")
    t.add_argument("--val_target", help="Directory for validation target masks")
    t.add_argument("--checkpoint_dir", help="Directory to save checkpoints")
    
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--input", required=True, help="Input images folder for validation")
    e.add_argument("--target", required=True, help="Target masks folder for validation")
    
    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", required=True)
    
    args = p.parse_args()

    if args.cmd == "train":
        run_train(
            csv_path=args.csv,
            train_input=args.train_input,
            train_target=args.train_target,
            val_input=args.val_input,
            val_target=args.val_target,
            checkpoint_dir=args.checkpoint_dir
        )
    elif args.cmd == "eval":
        run_eval(args.model, args.input, args.target)
    elif args.cmd == "infer":
        run_infer(args.model, args.input, args.output)
    else:
        p.print_help()
