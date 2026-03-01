"""
nnUNet_predict_noise - Inference with noise injection for dual-input trainers.

Tests whether each input channel (augmented / reference) contributes to segmentation
by replacing one with pure Gaussian noise N(0,1).

Noise Modes (clean-path evaluation — DualMask slot 1 output):
    0 = Standard inference: [img, img] (same as nnUNet_predict)
    1 = Noise in augmented slot: [noise, img]  - Tests if reference alone drives segmentation
    2 = Noise in reference slot: [img, noise]  - Tests if network can segment without reference

Noise Modes (aug-path evaluation — DualMask slot 0 output):
    3 = Noise in augmented slot: [noise, img]  - Aug-path output when augmented is noise
    4 = Noise in reference slot: [img, noise]  - Aug-path output when reference is noise

Usage:
    python -m nnunet.inference.predict_noise -i INPUT -o OUTPUT -t TASK -tr nnUNetTrainer_NoCLRamp --noise_mode 1 -f 0
"""

import argparse
import torch

from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir, save_json
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from time import time


def main():
    parser = argparse.ArgumentParser(
        description="nnUNet prediction with noise injection for dual-input models. "
                    "Replaces one input channel with Gaussian noise to test channel utility."
    )
    parser.add_argument("-i", '--input_folder', required=True,
                        help="Input folder with test cases (same format as nnUNet_predict)")
    parser.add_argument('-o', "--output_folder", required=True,
                        help="Folder for saving predictions")
    parser.add_argument('-t', '--task_name', required=True,
                        help='Task name or task ID')
    parser.add_argument('-tr', '--trainer_class_name', required=False, default=default_trainer,
                        help='Name of the nnUNetTrainer (e.g., nnUNetTrainer_NoCLRamp)')
    parser.add_argument('-m', '--model', default="3d_fullres", required=False,
                        help="2d, 3d_lowres, 3d_fullres. Default: 3d_fullres")
    parser.add_argument('-p', '--plans_identifier', default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="Folds to use for prediction. Default: auto-detect")

    # === NOISE MODE ===
    parser.add_argument('--noise_mode', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help="Noise injection mode: "
                             "0=standard [img,img] clean-path eval, "
                             "1=noise in aug slot (clean-path eval), "
                             "2=noise in ref slot (clean-path eval), "
                             "3=noise in aug slot (aug-path eval), "
                             "4=noise in ref slot (aug-path eval), "
                             "5=standard [img,img] aug-path eval")

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="Save softmax probabilities as npz")
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="Disable test time augmentation (mirroring)")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true")
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False)
    parser.add_argument("--step_size", type=float, default=0.5, required=False)
    parser.add_argument('-chk', default='model_final_checkpoint', required=False,
                        help='Checkpoint name, default: model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False)
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int)
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int)
    parser.add_argument("--part_id", type=int, required=False, default=0)
    parser.add_argument("--num_parts", type=int, required=False, default=1)

    args = parser.parse_args()

    # Parse task name
    task_name = args.task_name
    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    # Parse folds
    folds = args.folds
    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            parsed_folds = []
            for i in folds:
                try:
                    parsed_folds.append(int(i))
                except ValueError:
                    parsed_folds.append(i)
            folds = parsed_folds
    elif folds == "None":
        folds = None

    # Parse all_in_gpu
    all_in_gpu = args.all_in_gpu
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    # Build model folder path
    trainer = args.trainer_class_name
    model_folder_name = join(network_training_output_dir, args.model, task_name,
                             trainer + "__" + args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    # === SET NOISE MODE ON TRAINER ===
    # We use a module-level variable that trainers check during predict_preprocessed
    noise_mode = args.noise_mode
    noise_mode_names = {0: "standard [img, img] → clean-path", 
                        1: "noise in augmented [noise, img] → clean-path", 
                        2: "noise in reference [img, noise] → clean-path",
                        3: "noise in augmented [noise, img] → aug-path",
                        4: "noise in reference [img, noise] → aug-path",
                        5: "standard [img, img] → aug-path"}
    print(f"\n{'='*60}")
    print(f"NOISE MODE: {noise_mode} - {noise_mode_names[noise_mode]}")
    print(f"{'='*60}\n")

    # Set the noise mode as an environment-like global that trainers will pick up
    import nnunet.inference.predict_noise as noise_module
    noise_module.NOISE_MODE = noise_mode

    # Run prediction
    st = time()
    predict_from_folder(model_folder_name, args.input_folder, args.output_folder, folds,
                        args.save_npz, args.num_threads_preprocessing,
                        args.num_threads_nifti_save, None, args.part_id, args.num_parts,
                        not args.disable_tta,
                        overwrite_existing=args.overwrite_existing, mode=args.mode,
                        overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args.disable_mixed_precision,
                        step_size=args.step_size, checkpoint_name=args.chk)
    end = time()
    save_json(end - st, join(args.output_folder, 'prediction_time.txt'))

    print(f"\nPrediction complete. Noise mode: {noise_mode} ({noise_mode_names[noise_mode]})")
    print(f"Time: {end - st:.1f}s")
    print(f"Results saved to: {args.output_folder}")


# Module-level noise mode (set by CLI, read by trainers)
NOISE_MODE = 0


def get_noise_mode():
    """Get current noise mode. Called by trainers during predict_preprocessed."""
    try:
        import nnunet.inference.predict_noise as noise_module
        return getattr(noise_module, 'NOISE_MODE', 0)
    except (ImportError, AttributeError):
        return 0


if __name__ == "__main__":
    main()
