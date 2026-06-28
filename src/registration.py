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


def estimate_transformation(fixed, moving):
    ImageType = type(fixed)

    # 1: Rigit Registration
    RigidTransformType = itk.VersorRigid3DTransform[itk.D]
    rigid_transform = RigidTransformType.New()

    initializer = itk.CenteredTransformInitializer[
        RigidTransformType, ImageType, ImageType
    ].New()
    initializer.SetTransform(rigid_transform)
    initializer.SetFixedImage(fixed)
    initializer.SetMovingImage(moving)
    initializer.MomentsOn()
    initializer.InitializeTransform()

    rigid_transform = _check_transform(
        fixed,
        moving,
        rigid_transform,
        levels=3,
        shrink=[4, 2, 1],
        smooth=[2, 1, 0],
        learning_rate=1.0,
        min_step=0.001,
        iterations=100,
        stage_name="rigid",
    )

    # 2: Affine Registration
    AffineTransformType = itk.AffineTransform[itk.D, 3]
    affine_transform = AffineTransformType.New()
    affine_transform.SetIdentity()
    affine_transform.SetMatrix(rigid_transform.GetMatrix())
    affine_transform.SetOffset(rigid_transform.GetOffset())
    affine_transform.SetCenter(rigid_transform.GetCenter())

    affine_transform = _check_transform(
        fixed,
        moving,
        affine_transform,
        levels=2,
        shrink=[2, 1],
        smooth=[1, 0],
        learning_rate=0.5,
        min_step=0.0005,
        iterations=150,
        stage_name="affine",
    )

    return affine_transform
