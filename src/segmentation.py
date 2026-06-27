import itk

def tumor_segmentation(image, seed=[110, 100, 50], lower=180, upper=255, multiplier=3, iterations=5,
                   time_step=0.125):
    image_type = type(image)

    smoother = itk.GradientAnisotropicDiffusionImageFilter[image_type, image_type].New()
    smoother.SetInput(image)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetTimeStep(time_step)
    smoother.SetConductanceParameter(multiplier)
    smoother.Update()

    # # Instantiate the filter
    connected_threshold = itk.ConnectedThresholdImageFilter[image_type, image_type].New()

    # Configure filter
    connected_threshold.SetSeed(seed)
    connected_threshold.SetUpper(upper)
    connected_threshold.SetLower(lower)
    connected_threshold.SetInput(smoother.GetOutput())

    dimension = image.GetImageDimension()

    # Execute pipeline
    connected_threshold.Update()

    in_type = itk.output(connected_threshold)
    output_type = itk.Image[itk.UC, dimension]
    rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    return rescaler.GetOutputMaximum()
