import itk
import matplotlib.pyplot as plt


def segment(input_filepath="data/case6_gre1.nrrd",
         output_filepath='tumor-segmented.png', seedX=110, seedY=100, lower=180, upper=255):
    # Instantiate the reader
    input_image = itk.imread(input_filepath, pixel_type=itk.F)
    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(input_image), cmap="gray")
    plt.waitforbuttonpress()

    smoother = itk.GradientAnisotropicDiffusionImageFilter[type(input_image), type(input_image)].New()
    smoother.SetInput(input_image)
    smoother.SetNumberOfIterations(5)
    smoother.SetTimeStep(0.125)
    smoother.SetConductanceParameter(3)
    smoother.Update()


    # Display image with matplotlib
    plt.ion()
    plt.imshow(itk.GetArrayViewFromImage(smoother.GetOutput()) , cmap="gray")
    plt.waitforbuttonpress()
    #
    # # Instantiate the filter
    connected_threshold = itk.ConnectedThresholdImageFilter[type(input_image), type(input_image)].New()

    # Configure filter
    connected_threshold.SetSeed([100,110])
    connected_threshold.SetUpper(255)
    connected_threshold.SetLower(200)
    connected_threshold.SetInput(smoother.GetOutput())

    dimension = input_image.GetImageDimension()
    #
    # Execute pipeline
    connected_threshold.Update()
    result = connected_threshold.GetOutput()
    plt.imshow(itk.GetArrayViewFromImage(result), cmap="gray")
    plt.waitforbuttonpress()
    #
    in_type = itk.output(connected_threshold)
    output_type = itk.Image[itk.UC, dimension]
    rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    #
    itk.imwrite(rescaler, output_filepath)
