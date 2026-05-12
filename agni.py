"""
agni.py – Pipeline orchestrator (Stages 1 · 2 · 3)
=====================================================
Stage 1 : usable / unusable classification (trains on ALL labelled images)
Stage 2 : edge segmentation   – trains only on USABLE images
Stage 3 : edge refinement     – trains only on USABLE images
"""

import subprocess
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON   = sys.executable

STAGE1_SCRIPT = ROOT_DIR / "Stage1" / "stage1.py"
STAGE2_SCRIPT = ROOT_DIR / "Stage2" / "stage2.py"

STAGE3_SRC = ROOT_DIR / "Stage3" / "src"
sys.path.insert(0, str(ROOT_DIR))

from Stage3.src.inference import run_stage3


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd):
    print("\n" + "=" * 70)
    print("Running:", " ".join(str(c) for c in cmd))
    print("=" * 70)
    subprocess.run([str(c) for c in cmd], check=True)


def _ask(prompt, default=""):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _ask_range(default_min, default_max):
    if input(f"Use custom image range? [current: {default_min}–{default_max}] (y/n): ").strip().lower() != "y":
        return str(default_min), str(default_max)
    return (
        input(f"  MIN_IMG_NO (default {default_min}): ").strip() or str(default_min),
        input(f"  MAX_IMG_NO (default {default_max}): ").strip() or str(default_max),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli_args():
    p = argparse.ArgumentParser(description="Agni Pipeline (Stages 1 · 2 · 3)")
    p.add_argument("--stage",  choices=["1", "2", "3", "all"], help="Stage to run")
    p.add_argument("--mode",   choices=["train_infer", "infer"])
    p.add_argument("--input_dir",  help="Directory containing raw images")
    p.add_argument("--output_dir", help="Root directory for all results")
    p.add_argument("--mask_dir",   help="Ground-truth mask dir (Stage 2 training)")
    p.add_argument("--csv_path",   help="Stage 1 CSV (used by Stage 2/3 to filter usable)")

    # Stage 3 training dirs (optional – if you want to train Stage 3 as well)
    p.add_argument("--run_train_stage3", action="store_true")
    p.add_argument("--s3_train_input",  help="Stage 3 train input dir  (Stage 2 masks)")
    p.add_argument("--s3_train_target", help="Stage 3 train target dir (ground-truth edges)")
    p.add_argument("--s3_val_input",    help="Stage 3 val input dir")
    p.add_argument("--s3_val_target",   help="Stage 3 val target dir")

    # Hyperparameters
    p.add_argument("--s1_epochs",     type=int,   default=20)
    p.add_argument("--s1_batch_size", type=int,   default=16)
    p.add_argument("--s1_lr",         type=float, default=1e-4)
    p.add_argument("--s1_kfolds",     type=int,   default=10)

    p.add_argument("--s2_epochs",     type=int,   default=120)
    p.add_argument("--s2_batch_size", type=int,   default=4)
    p.add_argument("--s2_lr",         type=float, default=5e-5)
    p.add_argument("--s2_pos_weight", type=float, default=8.0)
    p.add_argument("--s2_bce_weight", type=float, default=0.75)

    p.add_argument("--s3_epochs",     type=int,   default=60)
    p.add_argument("--s3_batch_size", type=int,   default=24)
    p.add_argument("--s3_lr",         type=float, default=3e-3)

    p.add_argument("--min_img",  type=int, default=1)
    p.add_argument("--max_img",  type=int, default=20000)
    p.add_argument("--s1_model", help="Pretrained Stage 1 model path (infer mode)")
    p.add_argument("--s2_model", help="Pretrained Stage 2 model path (infer mode)")
    p.add_argument("--s3_model", help="Pretrained Stage 3 model path (infer mode)")

    return p.parse_args()


# ── Stage runners ──────────────────────────────────────────────────────────────

def _stage1(args, input_dir, stage1_out):
    stage1_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, STAGE1_SCRIPT,
        "--mode",       args.mode,
        "--input_dir",  str(input_dir),
        "--output_dir", str(stage1_out),
        "--min_img_no", str(args.min_img),
        "--max_img_no", str(args.max_img),
        "--epochs",     str(args.s1_epochs),
        "--batch_size", str(args.s1_batch_size),
        "--lr",         str(args.s1_lr),
        "--kfolds",     str(args.s1_kfolds),
    ]
    if args.mode == "infer" and args.s1_model:
        cmd += ["--model_path", args.s1_model]
    _run(cmd)


def _stage2(args, input_dir, stage2_out, stage1_csv):
    stage2_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, STAGE2_SCRIPT,
        "--mode",       args.mode,
        "--img_dir",    str(input_dir),
        "--output_dir", str(stage2_out),
        "--min_img_no", str(args.min_img),
        "--max_img_no", str(args.max_img),
        "--epochs",     str(args.s2_epochs),
        "--batch_size", str(args.s2_batch_size),
        "--lr",         str(args.s2_lr),
        "--pos_weight", str(args.s2_pos_weight),
        "--bce_weight", str(args.s2_bce_weight),
    ]
    if args.mode == "train_infer":
        if not args.mask_dir:
            raise ValueError("--mask_dir is required for Stage 2 training")
        cmd += [
            "--mask_dir",  str(Path(args.mask_dir).resolve()),
            "--csv_path",  str(stage1_csv),   # filters to usable-only
        ]
    else:
        s2_default = stage2_out / "checkpoints" / "edge_model_1.pth"
        model_path = args.s2_model or str(s2_default)
        cmd += ["--model_path", model_path]
    _run(cmd)


def _stage3(args, stage2_out, base_dir, stage1_csv):
    s2_masks   = stage2_out / "masks"
    stage3_out = base_dir / "Stage3_Output"
    ckpt_dir   = stage3_out / "checkpoints"

    # Optional: re-train Stage 3
    if args.run_train_stage3:
        from Stage3.src.train import run_stage3_training

        t_in  = args.s3_train_input  or str(s2_masks)
        t_tar = args.s3_train_target or input("Stage 3 Train Target Dir: ").strip()
        v_in  = args.s3_val_input    or t_in
        v_tar = args.s3_val_target   or t_tar

        run_stage3_training(
            train_input   = t_in,
            train_target  = t_tar,
            val_input     = v_in,
            val_target    = v_tar,
            checkpoint_dir= str(ckpt_dir),
            epochs        = args.s3_epochs,
            batch_size    = args.s3_batch_size,
            lr            = args.s3_lr,
            csv_path      = str(stage1_csv) if stage1_csv and stage1_csv.exists() else None,
        )

    # Stage 3 inference
    s3_default = ckpt_dir / "best_unet.pth"
    model_path = args.s3_model or (str(s3_default) if s3_default.exists() else None)

    run_stage3(
        input_mask_dir   = str(s2_masks),
        parent_output_dir= str(base_dir),
        model_path       = model_path,
    )


# ── Pipeline entry ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    mode       = args.mode
    input_dir  = Path(args.input_dir).resolve()  if args.input_dir  else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    base_dir   = output_dir / ("Training" if mode == "train_infer" else "Inference") if output_dir else None
    stage1_out = base_dir / "Stage1_Output" if base_dir else None
    stage2_out = base_dir / "Stage2_Output" if base_dir else None

    # CSV that Stage 1 produces (used by Stage 2 + 3 to filter usable images)
    stage1_csv = (
        stage1_out / "final_usable_unusable_images.csv"
        if stage1_out else
        Path(args.csv_path).resolve() if args.csv_path else None
    )

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.stage in ("1", "all"):
        if not input_dir or not output_dir:
            raise ValueError("--input_dir and --output_dir are required for Stage 1")
        _stage1(args, input_dir, stage1_out)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.stage in ("2", "all"):
        if not input_dir or not output_dir:
            raise ValueError("--input_dir and --output_dir are required for Stage 2")
        # If running standalone (not "all"), let user supply a csv
        csv = (
            stage1_csv
            if args.stage == "all"
            else Path(args.csv_path).resolve() if args.csv_path else stage1_csv
        )
        _stage2(args, input_dir, stage2_out, csv)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if args.stage in ("3", "all"):
        if not output_dir:
            raise ValueError("--output_dir is required for Stage 3")
        _stage3(args, stage2_out, base_dir, stage1_csv)


# ── Interactive mode ───────────────────────────────────────────────────────────

def interactive_mode():
    print("\n========== AGNI PIPELINE (Interactive) ==========\n")
    print("1) Stage 1 – Usable/Unusable Classification")
    print("2) Stage 2 – Edge Segmentation")
    print("3) Stage 1 + 2 + 3 – Full Pipeline")
    choice = input("Choice (1/2/3): ").strip()

    print("\n1) Training + Inference")
    print("2) Inference Only")
    mode = "train_infer" if input("Choice (1/2): ").strip() == "1" else "infer"

    # Build a minimal args-like object
    class Args:
        pass
    a = Args()
    a.mode = mode
    a.s1_epochs = 20; a.s1_batch_size = 16; a.s1_lr = 1e-4; a.s1_kfolds = 10
    a.s2_epochs = 120; a.s2_batch_size = 4; a.s2_lr = 5e-5
    a.s2_pos_weight = 8.0; a.s2_bce_weight = 0.75
    a.s3_epochs = 60; a.s3_batch_size = 24; a.s3_lr = 3e-3
    a.s1_model = a.s2_model = a.s3_model = None
    a.mask_dir = a.csv_path = None
    a.run_train_stage3 = False
    a.s3_train_input = a.s3_train_target = None
    a.s3_val_input   = a.s3_val_target   = None

    if choice == "1":
        a.stage      = "1"
        a.input_dir  = input("INPUT_DIR (images): ").strip()
        a.output_dir = input("OUTPUT_DIR (results): ").strip()
        mn, mx       = _ask_range(1, 20000)
        a.min_img    = int(mn); a.max_img = int(mx)
        if mode == "infer":
            a.s1_model = input("Stage 1 model path (blank = auto): ").strip() or None
        run_pipeline(a)
        return

    if choice == "2":
        a.stage      = "2"
        a.input_dir  = input("IMG_DIR (images): ").strip()
        a.output_dir = input("OUTPUT_DIR (results): ").strip()
        mn, mx       = _ask_range(19501, 20000)
        a.min_img    = int(mn); a.max_img = int(mx)
        if mode == "train_infer":
            a.mask_dir  = input("MASK_DIR: ").strip()
            a.csv_path  = input("Stage 1 CSV path: ").strip()
        else:
            a.s2_model  = input("Stage 2 model path (blank = auto): ").strip() or None
            a.csv_path  = input("Stage 1 CSV path (blank = skip filter): ").strip() or None
        run_pipeline(a)
        return

    if choice == "3":
        a.stage      = "all"
        a.input_dir  = input("INPUT_DIR (images): ").strip()
        a.output_dir = input("OUTPUT_DIR (results): ").strip()
        mn, mx       = _ask_range(1, 20000)
        a.min_img    = int(mn); a.max_img = int(mx)
        if mode == "train_infer":
            a.mask_dir = input("MASK_DIR for Stage 2: ").strip()
            train_s3 = input("Also train Stage 3? (y/n): ").strip().lower() == "y"
            a.run_train_stage3 = train_s3
            if train_s3:
                a.s3_train_input  = input("Stage 3 train input dir  (Stage 2 masks): ").strip() or None
                a.s3_train_target = input("Stage 3 train target dir (ground-truth edges): ").strip()
                a.s3_val_input    = input("Stage 3 val input dir (blank = same as train): ").strip() or None
                a.s3_val_target   = input("Stage 3 val target dir  (blank = same as train): ").strip() or None
        else:
            a.s1_model = input("Stage 1 model (blank = auto): ").strip() or None
            a.s2_model = input("Stage 2 model (blank = auto): ").strip() or None
            a.s3_model = input("Stage 3 model (blank = auto): ").strip() or None
            a.csv_path = input("Stage 1 CSV path (for usable filter, blank = skip): ").strip() or None
        run_pipeline(a)
        return

    print("Invalid choice.")


# ── Entry ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        args = _cli_args()
        run_pipeline(args)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
