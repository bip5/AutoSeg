import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_slice_max_area(mask, axis):
    """
    Find the slice index where the mask has the largest area (most non-zero pixels)
    along a specific axis.
    """
    if mask.sum() == 0:
        return mask.shape[axis] // 2

    best_count = -1
    best_idx = mask.shape[axis] // 2

    for idx in range(mask.shape[axis]):
        if axis == 0:
            count = np.count_nonzero(mask[idx, :, :])
        elif axis == 1:
            count = np.count_nonzero(mask[:, idx, :])
        else:
            count = np.count_nonzero(mask[:, :, idx])

        if count > best_count:
            best_count = count
            best_idx = idx

    return best_idx

def extract_slice(volume, axis, idx):
    if axis == 0:
        s = volume[idx, :, :]
    elif axis == 1:
        s = volume[:, idx, :]
        s = np.rot90(s, k=-1)  # Lateral view rotated -90 degrees
    else:
        s = volume[:, :, idx]
    return s

def load_summary(summary_path):
    """Load case-level dice scores from a summary.json file."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # Format: {"results": {"all": [{"reference": "case", ...}, ...]}}
    # We want case_name -> Dice
    case_metrics = {}
    if "results" in data and "all" in data["results"]:
        for item in data["results"]["all"]:
            # extract case name without extension
            # e.g., "sub-strokecase0001.nii.gz" -> "sub-strokecase0001"
            ref_name = Path(item["reference"]).name
            if ref_name.endswith(".nii.gz"):
                case_id = ref_name[:-7]
            else:
                case_id = ref_name

            # get the mean dice over classes, or the only class
            metrics = item["1"] if "1" in item else item.get("mean", {})
            dice = metrics.get("Dice", 0.0)
            case_metrics[case_id] = float(dice)
    return case_metrics

def overlay_mask(ax, img, mask, color, title, alpha=0.4):
    ax.imshow(img, cmap='gray', interpolation='none')
    if mask.sum() > 0:
        masked_data = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked_data, cmap=ListedColormap([color]), alpha=alpha, interpolation='none')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def plot_error_map(ax, img, gt, pred, title):
    """Plots FP as red, FN as blue, TP as green."""
    ax.imshow(img, cmap='gray', interpolation='none')
    
    # TP
    tp = (gt > 0) & (pred > 0)
    if tp.sum() > 0:
        ax.imshow(np.ma.masked_where(~tp, tp), cmap=ListedColormap(['green']), alpha=0.4, interpolation='none')
    
    # FP (pred but not GT)
    fp = (pred > 0) & (gt == 0)
    if fp.sum() > 0:
        ax.imshow(np.ma.masked_where(~fp, fp), cmap=ListedColormap(['red']), alpha=0.5, interpolation='none')
    
    # FN (GT but not pred)
    fn = (gt > 0) & (pred == 0)
    if fn.sum() > 0:
        ax.imshow(np.ma.masked_where(~fn, fn), cmap=ListedColormap(['blue']), alpha=0.5, interpolation='none')

    ax.set_title(title, fontsize=10)
    ax.axis('off')

def save_individual_plot(img, cmap, path, mask=None, mask_color=None, alpha=0.4):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap=cmap, interpolation='none')
    if mask is not None and mask.sum() > 0:
        masked_data = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked_data, cmap=ListedColormap([mask_color]), alpha=alpha, interpolation='none')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_individual_error_plot(img, gt, pred, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray', interpolation='none')
    
    tp = (gt > 0) & (pred > 0)
    if tp.sum() > 0:
        ax.imshow(np.ma.masked_where(~tp, tp), cmap=ListedColormap(['green']), alpha=0.4, interpolation='none')
    
    fp = (pred > 0) & (gt == 0)
    if fp.sum() > 0:
        ax.imshow(np.ma.masked_where(~fp, fp), cmap=ListedColormap(['red']), alpha=0.5, interpolation='none')
    
    fn = (gt > 0) & (pred == 0)
    if fn.sum() > 0:
        ax.imshow(np.ma.masked_where(~fn, fn), cmap=ListedColormap(['blue']), alpha=0.5, interpolation='none')

    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# --- MANIFEST LOGIC ---

def load_manifest(out_dir):
    manifest_path = Path(out_dir) / "vis_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {"single": {}, "compare": {}, "aggregate": {}}

def save_manifest(out_dir, manifest):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# --- SUBCOMMAND COMPUTATIONS ---

def do_single(args):
    out_dir = Path(args.out_dir)
    manifest = load_manifest(out_dir.parent if out_dir.parent.exists() else out_dir) # manifest lives in vis_base
    manifest_path_dir = out_dir.parent

    pred_dir = Path(args.pred_dir)
    pred_name = out_dir.name # e.g. baseline_300ep_f0

    summary_file = pred_dir / "summary.json"
    if not summary_file.exists():
        print(f"ERROR: No summary.json found in {pred_dir}")
        return

    case_metrics = load_summary(summary_file)
    n_cases = len(case_metrics)

    if not args.overwrite:
        if pred_name in manifest["single"]:
            if manifest["single"][pred_name].get("n_cases") == n_cases:
                print(f"SKIPPING: {pred_name} single visualization already generated.")
                return

    print(f"GENERATING single visualization for {pred_name} ({n_cases} cases)")

    # Sort cases by Dice ascending (worst first)
    sorted_cases = sorted(case_metrics.items(), key=lambda x: x[1])

    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)

    for rank, (case_id, dice) in enumerate(sorted_cases):
        case_out = out_dir / "cases" / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        # filename template {rank:03d}_{dice:.3f}_{case_id}.png
        prefix = f"{rank:03d}_{dice:.3f}_{case_id}"
        triptych_path = case_out / f"{prefix}_summary.png"

        # load volumes
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        img_path = img_dir / f"{case_id}_0000.nii.gz"

        if not pred_path.exists() or not gt_path.exists() or not img_path.exists():
            print(f"Warning: Missing files for {case_id}, skipping.")
            continue

        pred_vol = nib.load(str(pred_path)).get_fdata()
        gt_vol = nib.load(str(gt_path)).get_fdata()
        img_vol = nib.load(str(img_path)).get_fdata()

        # Name mapping for axes
        axis_names = {0: "Sagittal", 1: "Lateral", 2: "Axial"}

        for axis in range(3):
            axis_name = axis_names[axis]
            
            # Create root folder for composites
            axis_out_dir = out_dir / axis_name
            axis_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Triptych path directly in the root plane folder (surfaced)
            prefix = f"{rank:03d}_{dice:.3f}_{case_id}"
            triptych_path = axis_out_dir / f"{prefix}_summary.png"

            # Find best slice for this axis
            idx = get_slice_max_area(gt_vol, axis)

            pred_slice = extract_slice(pred_vol, axis, idx)
            gt_slice = extract_slice(gt_vol, axis, idx)
            img_slice = extract_slice(img_vol, axis, idx)

            # 1. Triptych
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_slice, cmap='gray', interpolation='none')
            axes[0].set_title(f"{case_id} ({axis_name})\nDice: {dice:.3f}", fontsize=10)
            axes[0].axis('off')

            overlay_mask(axes[1], img_slice, gt_slice, 'green', 'Ground Truth')
            overlay_mask(axes[2], img_slice, pred_slice, 'yellow', 'Prediction')
            plot_error_map(axes[3], img_slice, gt_slice, pred_slice, 'Error (FP:Red, FN:Blue)')

            plt.tight_layout()
            fig.savefig(triptych_path, bbox_inches='tight')
            plt.close(fig)

            # 2. Individual ones (tucked away)
            save_individual_plot(img_slice, 'gray', case_out / f"{axis_name}_image.png")
            save_individual_plot(img_slice, 'gray', case_out / f"{axis_name}_gt.png", gt_slice, 'green')
            save_individual_plot(img_slice, 'gray', case_out / f"{axis_name}_pred.png", pred_slice, 'yellow')
            save_individual_error_plot(img_slice, gt_slice, pred_slice, case_out / f"{axis_name}_error.png")

    # generate index contact sheet (just copy the top 20 worst paths, maybe overkill to merge all, let's just create an index html or simply a big plot)
    # Actually, a single giant plot with 40 rows might crash due to size. Let's skip the index.png and rely on file explorer sorting.
    
    manifest = load_manifest(manifest_path_dir)
    manifest["single"][pred_name] = {
        "completed": datetime.now().isoformat(),
        "n_cases": n_cases
    }
    save_manifest(manifest_path_dir, manifest)
    print("DONE.")


def do_compare(args):
    out_dir = Path(args.out_dir)
    manifest_dir = args.base_vis_dir if hasattr(args, 'base_vis_dir') else out_dir.parent
    manifest = load_manifest(manifest_dir)

    pred_a_dir = Path(args.pred_a)
    pred_b_dir = Path(args.pred_b)
    
    name_a = args.name_a
    name_b = args.name_b

    pair_key = tuple(sorted([name_a, name_b]))
    pair_str = f"{pair_key[0]}__vs__{pair_key[1]}"

    if not args.overwrite:
        if pair_str in manifest["compare"]:
            print(f"SKIPPING: {pair_str} comparison already generated.")
            return

    print(f"GENERATING comparison: {name_a} vs {name_b}")

    sum_a = load_summary(pred_a_dir / "summary.json") if (pred_a_dir / "summary.json").exists() else {}
    sum_b = load_summary(pred_b_dir / "summary.json") if (pred_b_dir / "summary.json").exists() else {}

    common_cases = set(sum_a.keys()).intersection(set(sum_b.keys()))

    # Sort by absolute difference in Dice (biggest difference first)
    diffs = []
    for case in common_cases:
        dice_a = sum_a[case]
        dice_b = sum_b[case]
        diffs.append((case, dice_a, dice_b, abs(dice_a - dice_b)))
    
    diffs.sort(key=lambda x: x[3], reverse=True)

    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)

    comp_dir = out_dir / pair_str / "cases"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for rank, (case_id, dice_a, dice_b, delta) in enumerate(diffs):
        prefix = f"{rank:03d}_{delta:.3f}_{case_id}"
        comp_path = comp_dir / f"{prefix}_comparison.png"

        gt_path = gt_dir / f"{case_id}.nii.gz"
        img_path = img_dir / f"{case_id}_0000.nii.gz"
        a_path = pred_a_dir / f"{case_id}.nii.gz"
        b_path = pred_b_dir / f"{case_id}.nii.gz"

        if not all([p.exists() for p in [gt_path, img_path, a_path, b_path]]):
            continue

        gt_vol = nib.load(str(gt_path)).get_fdata()
        img_vol = nib.load(str(img_path)).get_fdata()
        a_vol = nib.load(str(a_path)).get_fdata()
        b_vol = nib.load(str(b_path)).get_fdata()

        # Name mapping for axes
        axis_names = {0: "Sagittal", 1: "Lateral", 2: "Axial"}

        for axis in range(3):
            axis_name = axis_names[axis]
            
            # Create root folder for composites
            axis_out_dir = comp_dir / axis_name
            axis_out_dir.mkdir(parents=True, exist_ok=True)
            
            comp_path = axis_out_dir / f"{prefix}_comparison.png"

            idx = get_slice_max_area(gt_vol, axis)
            
            gt_s = extract_slice(gt_vol, axis, idx)
            img_s = extract_slice(img_vol, axis, idx)
            a_s = extract_slice(a_vol, axis, idx)
            b_s = extract_slice(b_vol, axis, idx)

            # 6 panels: Image | GT | Pred A | Pred B | Err A | Err B
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            
            axes[0].imshow(img_s, cmap='gray', interpolation='none')
            axes[0].set_title(f"{case_id} ({axis_name})\nΔ: {delta:.3f}", fontsize=10)
            axes[0].axis('off')

            overlay_mask(axes[1], img_s, gt_s, 'green', 'Ground Truth')
            overlay_mask(axes[2], img_s, a_s, 'yellow', f"Pred A\n{name_a} (*{dice_a:.3f}*)")
            overlay_mask(axes[3], img_s, b_s, 'yellow', f"Pred B\n{name_b} (*{dice_b:.3f}*)")

            plot_error_map(axes[4], img_s, gt_s, a_s, f"Error A")
            plot_error_map(axes[5], img_s, gt_s, b_s, f"Error B")

            plt.tight_layout()
            fig.savefig(comp_path, bbox_inches='tight')
            plt.close(fig)

    manifest = load_manifest(manifest_dir)
    manifest["compare"][pair_str] = {
        "completed": datetime.now().isoformat()
    }
    save_manifest(manifest_dir, manifest)
    print("DONE comparison.")


def do_aggregate(args):
    # Find all models in base_dir with matching prefixes
    base_dir = Path(args.base_dir)
    prefixes = args.prefixes

    models = {}
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if any(entry.name.startswith(p) for p in prefixes):
            sum_path = entry / "summary.json"
            if sum_path.exists():
                models[entry.name] = load_summary(sum_path)
    
    if not models:
        print("No models found for aggregation.")
        return

    out_dir = Path(args.out_dir) / "aggregate"
    manifest_dir = Path(args.out_dir)
    manifest = load_manifest(manifest_dir)

    model_list_sorted = sorted(list(models.keys()))
    
    if not args.overwrite:
        if "aggregate" in manifest and manifest["aggregate"].get("models") == model_list_sorted:
            print("SKIPPING: Aggregate already generated for exact model list.")
            return

    print(f"GENERATING aggregate for {len(model_list_sorted)} models...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    data = []
    labels = []
    
    for m in model_list_sorted:
        dices = list(models[m].values())
        data.append(dices)
        labels.append(m)

    ax.boxplot(data, labels=labels)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Dice Score")
    ax.set_title("Dice Distribution across Models")
    plt.tight_layout()
    fig.savefig(out_dir / "dice_boxplot.png", dpi=150)
    plt.close(fig)

    # 2. Scatter plot (Volume vs Dice) - Needs volumes, which requires gt_dir
    # To keep it fast, we can approximate volume from the number of foreground voxels in GT if we load them
    # OR we skip this for now to keep the script fast. Let's do a variation of the scatter:
    # We will just plot Case ID vs Dice
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, m in enumerate(model_list_sorted):
        items = list(models[m].items())
        # Sort items by case_id for consistency
        items.sort(key=lambda x: x[0])
        # X is just 0...N
        x = np.arange(len(items))
        y = [v[1] for v in items]
        ax.scatter(x, y, label=m, alpha=0.6, s=15)
    
    ax.set_title("Dice per Case across Models")
    ax.set_ylabel("Dice Score")
    ax.set_xlabel("Case Index")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_dice.png", dpi=150)
    plt.close(fig)

    # 3. Auto-generate comparisons for all pairs
    import itertools
    for pair in itertools.combinations(model_list_sorted, 2):
        class FakeArgs:
            pass
        ca = FakeArgs()
        ca.pred_a = base_dir / pair[0]
        ca.pred_b = base_dir / pair[1]
        ca.name_a = pair[0]
        ca.name_b = pair[1]
        ca.gt_dir = args.gt_dir
        ca.img_dir = args.img_dir
        ca.out_dir = Path(args.out_dir) / "compare"
        ca.overwrite = args.overwrite
        ca.base_vis_dir = manifest_dir
        do_compare(ca)

    manifest = load_manifest(manifest_dir)
    manifest["aggregate"] = {
        "completed": datetime.now().isoformat(),
        "models": model_list_sorted
    }
    save_manifest(manifest_dir, manifest)
    print("DONE aggregate.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # single
    p_single = subparsers.add_parser('single')
    p_single.add_argument('--pred-dir', required=True)
    p_single.add_argument('--gt-dir', required=True)
    p_single.add_argument('--img-dir', required=True)
    p_single.add_argument('--out-dir', required=True)
    p_single.add_argument('--overwrite', action='store_true')

    # compare
    p_compare = subparsers.add_parser('compare')
    p_compare.add_argument('--pred-a', required=True)
    p_compare.add_argument('--pred-b', required=True)
    p_compare.add_argument('--name-a', required=True)
    p_compare.add_argument('--name-b', required=True)
    p_compare.add_argument('--gt-dir', required=True)
    p_compare.add_argument('--img-dir', required=True)
    p_compare.add_argument('--out-dir', required=True)
    p_compare.add_argument('--overwrite', action='store_true')

    # aggregate
    p_agg = subparsers.add_parser('aggregate')
    p_agg.add_argument('--base-dir', required=True)
    p_agg.add_argument('--prefixes', nargs='+', required=True)
    p_agg.add_argument('--gt-dir', required=True)
    p_agg.add_argument('--img-dir', required=True)
    p_agg.add_argument('--out-dir', required=True)
    p_agg.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    if args.command == 'single':
        do_single(args)
    elif args.command == 'compare':
        do_compare(args)
    elif args.command == 'aggregate':
        do_aggregate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
