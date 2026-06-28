import itk


def _check_transform(
    fixed,
    moving,
    initial_transform,
    levels: int,
    shrink: list[int],
    smooth: list[int],
    learning_rate: float,
    min_step: float,
    iterations: int,
    stage_name: str,
):
    ImageType = type(fixed)

    optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
    optimizer.SetLearningRate(learning_rate)
    optimizer.SetMinimumStepLength(min_step)
    optimizer.SetRelaxationFactor(0.5)
    optimizer.SetNumberOfIterations(iterations)

    metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
    metric.SetNumberOfHistogramBins(50)

    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New(
        FixedImage=fixed,
        MovingImage=moving,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=initial_transform,
    )

    scales_estimator = itk.RegistrationParameterScalesFromPhysicalShift[
        type(metric)
    ].New()
    scales_estimator.SetMetric(metric)
    optimizer.SetScalesEstimator(scales_estimator)

    registration.SetNumberOfLevels(levels)
    registration.SetShrinkFactorsPerLevel(shrink)
    registration.SetSmoothingSigmasPerLevel(smooth)
    registration.InPlaceOn()

    registration.Update()

    print(f"  [{stage_name}] stop condition: {optimizer.GetStopConditionDescription()}")
    print(
        f"  [{stage_name}] final metric value (Mattes MI, lower=better): {optimizer.GetValue():.4f}"
    )

    return initial_transform
