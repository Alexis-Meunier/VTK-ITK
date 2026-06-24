import itk
import matplotlib.pyplot as plt

def recalage_translation(fixed_filepath="data/case6_gre1.nrrd",
         moving_filepath="data/case6_gre2.nrrd", output_filepath='recalage-translation.png'):
    PixelType = itk.F
    fixed_image = itk.imread(fixed_filepath, PixelType)
    moving_image = itk.imread(moving_filepath, PixelType)

    dimension = fixed_image.GetImageDimension()
    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.TranslationTransform[itk.D, dimension]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
    optimizer.SetNumberOfIterations(10)


    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType,MovingImageType].New()
    fixed_interpolation = itk.LinearInterpolateImageFunction[FixedImageType, itk.D].New()
    metric.SetFixedInterpolator(fixed_interpolation)


    registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New()


    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetOptimizer(optimizer)
    registration.SetMetric(metric)
    registration.SetInitialTransform(initialTransform)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()
    translation_along_x = final_parameters.GetElement(0)
    translation_along_y = final_parameters.GetElement(1)

    number_of_iterations = optimizer.GetCurrentIteration()

    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Translation X = " + str(translation_along_x))
    print(" Translation Y = " + str(translation_along_y))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))


    resampler = itk.ResampleImageFilter[FixedImageType, MovingImageType].New()
    resampler.SetDefaultPixelValue(1)

    resampler.SetInterpolator(fixed_interpolation)
    resampler.SetInput(moving_image)
    resampler.SetTransform(transform)
    resampler.UseReferenceImageOn()
    resampler.SetReferenceImage(fixed_image)
    #
    subtraction = itk.SubtractImageFilter(Input1=fixed_image, Input2=resampler)
    subtraction.Update()
    plt.ion()
    plt.imshow(fixed_image)
    plt.imshow(itk.GetArrayViewFromImage(resampler))
    plt.waitforbuttonpress()

    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(subtraction))
    plt.waitforbuttonpress()

    OutputPixelType = itk.ctype('unsigned char')
    OutputImageType = itk.Image[OutputPixelType, dimension]
    caster = itk.CastImageFilter[FixedImageType, OutputImageType].New(resampler)

    itk.imwrite(caster, output_filepath)

def recalage_transformation_rigide(fixed_filepath="data/case6_gre1.nrrd",
         moving_filepath="data/case6_gre2.nrrd", output_filepath='recalage-rigide.png'):
    PixelType = itk.ctype('float')
    fixed_image = itk.imread(fixed_filepath, PixelType)
    moving_image = itk.imread(moving_filepath, PixelType)

    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.Rigid2DTransform
    initialTransform = TransformType.New()

    fixed_parameters = initialTransform.GetFixedParameters()
    fixed_parameters[0] = 0
    fixed_parameters[1] = 0
    initialTransform.SetFixedParameters(fixed_parameters)

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)

    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()
    fixed_interpolation = itk.LinearInterpolateImageFunction[FixedImageType, itk.D].New()
    metric.SetFixedInterpolator(fixed_interpolation)

    registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixed_image, MovingImage=moving_image, Metric=metric,
                                                     Optimizer=optimizer, InitialTransform=initialTransform)

    # Set the scales
    scale_parameters = initialTransform.GetParameters()
    scale_parameters[0] = 1000
    scale_parameters[1] = 1
    scale_parameters[2] = 1
    optimizer.SetScales(scale_parameters)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()
    angle = final_parameters.GetElement(0)
    translation_along_x = final_parameters.GetElement(1)
    translation_along_y = final_parameters.GetElement(2)

    number_of_iterations = optimizer.GetCurrentIteration()

    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Angle = " + str(angle))
    print(" Translation X = " + str(translation_along_x))
    print(" Translation Y = " + str(translation_along_y))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))

    plt.ion()
    plt.imshow(moving_image)
    plt.waitforbuttonpress()

    resampler = itk.ResampleImageFilter.New(Input=moving_image, Transform=transform, UseReferenceImage=True,
                                            ReferenceImage=fixed_image)
    resampler.SetDefaultPixelValue(1)
    resampler.Update()

    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(resampler))
    plt.waitforbuttonpress()
    subtraction = itk.SubtractImageFilter(Input1=fixed_image, Input2=resampler)
    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(subtraction))
    plt.waitforbuttonpress()

def recalage_information_mutuelle(fixed_filepath="data/case6_gre1.nrrd",
         moving_filepath="data/case6_gre2.nrrd", output_filepath="recalage-mutuelle.png", ):
    PixelType = itk.ctype("float")
    fixed_image = itk.imread(fixed_filepath, PixelType)
    moving_image = itk.imread(moving_filepath, PixelType)

    dimension = fixed_image.GetImageDimension()
    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.Rigid2DTransform[itk.D]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()

    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)

    metric = itk.MattesMutualInformationImageToImageMetricv4[FixedImageType, MovingImageType].New()
    metric.SetNumberOfHistogramBins(10)

    registration = itk.ImageRegistrationMethodv4.New()
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(initialTransform)


    # Set the scales
    scale_parameters = initialTransform.GetParameters()
    scale_parameters[0] = 1000
    scale_parameters[1] = 1
    scale_parameters[2] = 1
    optimizer.SetScales(scale_parameters)


    # Set the center of the image
    fixed_parameters = initialTransform.GetFixedParameters()
    fixed_parameters[0] = moving_image.GetLargestPossibleRegion().GetSize()[0] / 2.0
    fixed_parameters[1] = moving_image.GetLargestPossibleRegion().GetSize()[1] / 2.0

    initialTransform.SetFixedParameters(fixed_parameters)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()
    angle = final_parameters.GetElement(0)
    translation_along_x = final_parameters.GetElement(1)
    translation_along_y = final_parameters.GetElement(2)

    number_of_iterations = optimizer.GetCurrentIteration()

    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Angle = " + str(angle))
    print(" Translation X = " + str(translation_along_x))
    print(" Translation Y = " + str(translation_along_y))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))

    resampler = itk.ResampleImageFilter.New(Input=moving_image, Transform=transform, UseReferenceImage=True,
                                            ReferenceImage=fixed_image, )
    resampler.SetDefaultPixelValue(1)

    subtraction = itk.SubtractImageFilter(Input1=fixed_image, Input2=resampler)
    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(subtraction))
    plt.waitforbuttonpress()

    OutputPixelType = itk.ctype("unsigned char")
    OutputImageType = itk.Image[OutputPixelType, dimension]
    caster = itk.CastImageFilter[FixedImageType, OutputImageType].New(resampler)

    itk.imwrite(caster, output_filepath)
