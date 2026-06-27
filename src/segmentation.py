import itk

def tumor_segmentation(image, seed=[110, 100, 50], multiplier=1.5, iterations=5, initial_radius=2):
    mask = itk.confidence_connected_image_filter(
        image,
        seed=seed,
        multiplier=multiplier,
        number_of_iterations=iterations,
        initial_neighborhood_radius=initial_radius,
        replace_value=1,
    )
    return mask

