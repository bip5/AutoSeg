"""
Patch script to populate validation_raw for existing CLRamp/NoCLRamp training runs.

Usage:
    python patch_validation_raw.py -t TASK_ID -tr TRAINER_NAME [-f FOLDS] [-m MODEL]

Example:
    python patch_validation_raw.py -t 501 -tr nnUNetTrainer_CLRamp -f 0 1 2 3 4 -m 3d_fullres

This script loads each fold's best checkpoint and runs full validation to populate
the validation_raw/ folder, enabling nnUNet_find_best_configuration to work.
"""

import argparse
import os
from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile

from nnunet.paths import network_training_output_dir
from nnunet.training.model_restore import load_best_model_for_inference
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    parser = argparse.ArgumentParser(description="Populate validation_raw for existing training runs")
    parser.add_argument("-t", "--task", type=int, required=True, help="Task ID (e.g., 501)")
    parser.add_argument("-tr", "--trainer", type=str, required=True, help="Trainer class name")
    parser.add_argument("-f", "--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Folds to process")
    parser.add_argument("-m", "--model", type=str, default="3d_fullres", help="Model type (2d, 3d_fullres, etc.)")
    parser.add_argument("-p", "--plans", type=str, default="nnUNetPlansv2.1", help="Plans identifier")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing validation_raw")
    
    args = parser.parse_args()
    
    task_name = convert_id_to_task_name(args.task)
    print(f"\nPatching validation_raw for {task_name}")
    print(f"Trainer: {args.trainer}")
    print(f"Model: {args.model}")
    print(f"Folds: {args.folds}\n")
    
    base_folder = join(network_training_output_dir, args.model, task_name, args.trainer + "__" + args.plans)
    
    if not isdir(base_folder):
        raise RuntimeError(f"Training folder not found: {base_folder}")
    
    for fold in args.folds:
        fold_folder = join(base_folder, f"fold_{fold}")
        validation_folder = join(fold_folder, "validation_raw")
        
        print(f"\n{'='*50}")
        print(f"Processing fold {fold}")
        print(f"{'='*50}")
        
        if not isdir(fold_folder):
            print(f"  WARNING: Fold folder not found, skipping: {fold_folder}")
            continue
        
        # Check if validation_raw already has content
        if isdir(validation_folder) and not args.overwrite:
            existing_files = [f for f in os.listdir(validation_folder) if f.endswith('.nii.gz')]
            if existing_files:
                print(f"  validation_raw already has {len(existing_files)} files. Use --overwrite to regenerate.")
                continue
        
        # Load the best model for this fold
        checkpoint_file = join(fold_folder, "model_best.model")
        if not isfile(checkpoint_file):
            checkpoint_file = join(fold_folder, "model_final_checkpoint.model")
        
        if not isfile(checkpoint_file):
            print(f"  WARNING: No checkpoint found in {fold_folder}")
            continue
        
        print(f"  Loading checkpoint: {checkpoint_file}")
        
        # Use model_restore to load the trainer
        from nnunet.training.model_restore import restore_model
        trainer = restore_model(join(fold_folder, "model_best.model.pkl"), checkpoint=checkpoint_file, train=False)
        
        print(f"  Running full validation...")
        trainer.validate(
            do_mirroring=True,
            use_sliding_window=True,
            step_size=0.5,
            save_softmax=True,
            use_gaussian=True,
            overwrite=args.overwrite,
            validation_folder_name='validation_raw',
            run_postprocessing_on_folds=True
        )
        
        print(f"  ✓ Validation complete for fold {fold}")
    
    print(f"\n{'='*50}")
    print("Done! You can now run nnUNet_find_best_configuration")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
