import os
import itk

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
    render_3d_comparison
)

OUTPUT_DIR = "output"

SCAN1_PATH = "data/case6_gre1.nrrd"
SCAN2_PATH = "data/case6_gre2.nrrd"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fixed = itk.imread(SCAN1_PATH, itk.F)
    moving = itk.imread(SCAN2_PATH, itk.F)

    transform = estimate_transformation(fixed, moving)
    itk.transformwrite([transform], f"{OUTPUT_DIR}/affine_transform.tfm")

    mask1 = tumor_segmentation(fixed, SEED_SCAN1)
    mask2 = tumor_segmentation(moving, SEED_SCAN2)
    mask2_registered = apply_transformation(mask2, fixed, transform)

    metrics = compute_change_metrics(mask1, mask2_registered, fixed)
    print_summary(metrics)

    plot_2d_comparison(
        fixed, metrics["mask1_array"], metrics["mask2_array"],
        f"{OUTPUT_DIR}/2d_comparison.png",
    )
    render_3d_comparison(
        metrics["mask1_array"], metrics["mask2_array"],
        f"{OUTPUT_DIR}/3d_render.png",
        interactive=True,
    )


if __name__ == "__main__":
    main()
