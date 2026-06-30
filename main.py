import os
import itk
import argparse

from src.registration import (
    estimate_transformation,
    apply_transformation,
)

from src.segmentation import (
    tumor_segmentation,
    SEED_SCAN1,
    SEED_SCAN2,
)

from src.analysis import (
    compute_change_metrics,
    print_summary,
)

from src.visualization import (
    plot_2d_comparison,
    render_3d_comparison,
    interactive_slice_viewer,
    render_volume_raycast,
)

OUTPUT_DIR = "output"

SCAN1_PATH = "data/case6_gre1.nrrd"
SCAN2_PATH = "data/case6_gre2.nrrd"

parser = argparse.ArgumentParser()
parser.add_argument('--display', choices=['none', 'slice', 'comparison', 'volume'], default='slice', help='Type of visualization to show')
parser.add_argument('--plot', action='store_true')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args = parser.parse_args()

    print("[1/5] Loading scans...")
    fixed = itk.imread(SCAN1_PATH, itk.F)
    moving = itk.imread(SCAN2_PATH, itk.F)

    print("[2/5] Registering scan 2 onto scan 1 (rigid -> affine)...")
    transform = estimate_transformation(fixed, moving)
    itk.transformwrite([transform], f"{OUTPUT_DIR}/affine_transform.tfm")

    print("[3/5] Segmenting the tumor in both scans...")
    mask1 = tumor_segmentation(fixed, SEED_SCAN1)
    mask2 = tumor_segmentation(moving, SEED_SCAN2)

    mask2_registered = apply_transformation(mask2, fixed, transform)

    print("[4/5] Computing change metrics...")
    metrics = compute_change_metrics(mask1, mask2_registered, fixed)
    print_summary(metrics)

    print("[5/5] Generating visualizations...")
    if args.plot or args.display == 'none':
        plot_2d_comparison(
            fixed, metrics["mask1_array"], metrics["mask2_array"],
            f"{OUTPUT_DIR}/2d_comparison.png",
        )
    if args.display == 'slice':
        interactive_slice_viewer(
            fixed,
            metrics["mask1_array"],
            metrics["mask2_array"],
            f"{OUTPUT_DIR}/slice_viewer.png"
        )
    elif args.display == 'volume':
        render_volume_raycast(
            fixed,
            metrics["mask1_array"],
            f"{OUTPUT_DIR}/volume_render.png",
            interactive=True,
        )
    elif args.display == 'comparison':
        render_3d_comparison(
            metrics["mask1_array"],
            metrics["mask2_array"],
            f"{OUTPUT_DIR}/3d_render.png",
            interactive=True,
        )

    print("\nDone. Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
