import itk

# Hard coded values because subject asked us
# "Ce script devra s’exécuter sans nécessiter d’interaction utilisateur"
SEED_SCAN1 = [103, 77, 51]
SEED_SCAN2 = [91, 89, 53]

# region growing parameters, same for both scans so the two
# segmentations are produced by the exatc same method
MULTIPLIER = 1.2
NUM_ITERATIONS = 2
RADIUS = 2


def tumor_segmentation(image, seed, multiplier=MULTIPLIER, iterations=NUM_ITERATIONS,
                   initial_radius=RADIUS):
    """
    Segment the tumor in `image` starting using a region growing algorithm.
    The region starts at coordinates `seed`.

    Returns a mask representing the segmented tumor (1 = tumor, 0 = background).
    """
    mask = itk.confidence_connected_image_filter(
        image,
        seed=seed,
        multiplier=multiplier,
        number_of_iterations=iterations,
        initial_neighborhood_radius=initial_radius,
        replace_value=1,
    )
    return mask
