import itk
import vtk
import os

def affichage_lisse():
    # Read the data from a SLC file
    filePath = os.path.join(os.path.dirname(__file__), "../Data/cow.vtk")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filePath)

    # Triangulate
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(reader.GetOutputPort())

    # Decimate
    deci = vtk.vtkDecimatePro()
    deci.SetInputConnection(tri.GetOutputPort())
    deci.SetTargetReduction(0.3)
    deci.PreserveTopologyOn()
    deci.SetMaximumError(0.0002)

    # Smooth
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(deci.GetOutputPort())
    smooth.SetNumberOfIterations(25)
    smooth.SetRelaxationFactor(0.05)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())

    # Rendering pipeline
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(255 / 255, 192 / 255, 203 / 255)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()) # More "natural" interaction style

    render_window.Render()
    render_window_interactor.Start()


def create_triangle(
    p1: tuple[float, float, float]=(0.0, 0.0, 0.0),
    p2: tuple[float, float, float]=(1.0, 0.0, 0.0),
    p3: tuple[float, float, float]=(0.0, 1.0, 0.0),
):
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(3)
    points.SetPoint(0, *p1)
    points.SetPoint(1, *p2)
    points.SetPoint(2, *p3)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(3)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(1)
    cells.InsertCellPoint(2)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Rendering pipeline
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()) # More "natural" interaction style

    render_window.Render()
    render_window_interactor.Start()


def rendu_volumique():
    # Create the renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)
    iren.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()) # More "natural" interaction style

    # Read the data from a SLC file
    filePath = os.path.join(os.path.dirname(__file__), "../Data/poship.slc")
    reader = vtk.vtkSLCReader()
    reader.SetFileName(filePath)

    # Create transfer functions for opacity and color
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(20, 0.0)
    opacity_transfer_function.AddPoint(255, 0.3)

    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    color_transfer_function.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    color_transfer_function.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

    # Create properties, mappers, volume actors, and ray cast function
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)

    # Create the volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())

    # Create the volume and set the mapper and property
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Add this volume to the renderer and get a closer look
    renderer.AddActor(volume)
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    renderer.SetBackground(0.1, 0.1, 0.3)


    render_window.Render()

    # Interact with the data at 3 frames per second
    iren.SetDesiredUpdateRate(5.0)
    # iren.SetStillUpdateRate(0.001)
    iren.Start()


def image3D_contour():
    # You'll need to read the file
    reader = vtk.vtkXMLImageDataReader()

    # You'll need a render window to show it in
    renwin = vtk.vtkRenderWindow()

    # You'll need a renderer in the render window
    renderer = vtk.vtkRenderer()

    # Set the file name on the reader to filename.c_str()
    filePath = os.path.join(os.path.dirname(__file__), "../Data/head.vti")
    reader.SetFileName(filePath)

    # Put in renderer in the render window
    renwin.AddRenderer(renderer)

    # Create the interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Set the interactor on the render window
    renwin.SetInteractor(interactor)

    # Create a vtk.vtkContourFilter to do the isocontouring
    # Set the input to the output of the reader
    # Set the value to 135
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0, 135)

    # This is the vtk.vtkPolyDataMapper
    contourMapper = vtk.vtkPolyDataMapper()

    # Connect the mapper to the contour filter
    # Remember to turn ScalarVisibilityOff()
    contourMapper.SetInputConnection(contour.GetOutputPort())
    contourMapper.ScalarVisibilityOff()

    # This is the vtk.vtkActor
    contourActor = vtk.vtkActor()

    # Set the mapper
    contourActor.SetMapper(contourMapper)

    # Add the actor
    renderer.AddActor(contourActor)

    # Render and start the interactor
    renwin.Render()
    interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()) # More "natural" interaction style
    interactor.Start()


def image3D_volumique():
    # You'll need to read the file
    reader = vtk.vtkXMLImageDataReader()

    # You'll need a render window to show it in
    renwin = vtk.vtkRenderWindow()

    # You'll need a renderer in the render window
    renderer = vtk.vtkRenderer()

    # Set the file name on the reader to filename.c_str()
    filePath = os.path.join(os.path.dirname(__file__), "../Data/head.vti")
    reader.SetFileName(filePath)

    # Put in renderer in the render window
    renwin.AddRenderer(renderer)

    # Create the interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Set the interactor on the render window
    renwin.SetInteractor(interactor)

    # Create a vtk.vtkContourFilter to do the isocontouring
    # Set the input to the output of the reader
    # Set the value to 135
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0, 135)

    # This is the vtk.vtkPolyDataMapper
    contourMapper = vtk.vtkPolyDataMapper()

    # Connect the mapper to the contour filter
    # Remember to turn ScalarVisibilityOff()
    contourMapper.SetInputConnection(contour.GetOutputPort())
    contourMapper.ScalarVisibilityOff()

    # This is the vtk.vtkActor
    contourActor = vtk.vtkActor()

    # Set the mapper
    contourActor.SetMapper(contourMapper)

    # Add the actor
    renderer.AddActor(contourActor)

    # Create an opacity transfer function to map
    # scalar value to opacity
    opacityFun = vtk.vtkPiecewiseFunction()

    # Set a mapping going from 0.0 opacity at 90, up to 0.2 at 100,
    # and back down to 0.0 at 120.
    opacityFun.AddPoint(0.0, 0.0)
    opacityFun.AddPoint(90.0, 0.0)
    opacityFun.AddPoint(100.0, 0.2)
    opacityFun.AddPoint(120.0, 0.0)

    # Create a color transfer function for the mapping of scalar
    # value into color
    colorFun = vtk.vtkColorTransferFunction()

    # Set the color to a constant value, you might
    # want to try (0.8, 0.4, 0.2)
    colorFun.AddRGBPoint(0.0, .8, .4, .2)
    colorFun.AddRGBPoint(255.0, .8, .4, .2)

    # Create a volume property
    # Set the opacity and color. Change interpolation
    # to linear for a more pleasing image
    property = vtk.vtkVolumeProperty()
    property.SetScalarOpacity(opacityFun)
    property.SetColor(colorFun)
    property.SetInterpolationTypeToLinear()

    # Create the volume mapper
    mapper = vtk.vtkSmartVolumeMapper()

    # Set the input to the output of the reader
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create the volume
    volume = vtk.vtkVolume()

    # Set the property and the mapper
    volume.SetProperty(property)
    volume.SetMapper(mapper)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Render and start the interactor
    renwin.Render()
    interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()) # More "natural" interaction style
    interactor.Start()


def setupCamera(renderer, imageSlice):
    """Configure active camera of renderer by fitting the data"""

    camera = renderer.GetActiveCamera()
    renderer.ResetCamera()
    camera.ParallelProjectionOn()

    source = imageSlice.GetMapper().GetInput()

    extent = source.GetExtent()
    origin = source.GetOrigin()
    spacing = source.GetSpacing()

    xcenter = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    ycenter = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    zcenter = origin[2] + 0.5 * (extent[4] + extent[5]) * spacing[2]
    # xdimension = (extent[1] - extent[0] + 1) * spacing[0]
    ydimension = (extent[3] - extent[2] + 1) * spacing[1]
    # zdimension = (extent[5] - extent[4] + 1) * spacing[2]

    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * ydimension)
    camera.SetFocalPoint(xcenter, ycenter, zcenter)
    camera.SetPosition(xcenter, ycenter, zcenter - d)
    camera.SetViewUp(0, -1, 0)

    renderer.ResetCameraClippingRange()

def render(itkImage):
    """ Renders an itk Image with VTK """

    source = itk.vtk_image_from_image(itkImage)

    renderWindow = vtk.vtkRenderWindow()

    renderer = vtk.vtkRenderer()
    renderer.GetActiveCamera().ParallelProjectionOn()

    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)# More "natural" interaction style

    imageStyle = vtk.vtkInteractorStyleImage()
    imageStyle.SetInteractionModeToImageSlicing()
    renderWindowInteractor.SetInteractorStyle(imageStyle) # Interactor style for images, no rotation. Left click changes the window level of the image

    imageSliceMapper = vtk.vtkImageSliceMapper()
    imageSliceMapper.SetInputData(source)
    imageSliceMapper.SetSliceAtFocalPoint(True)
    imageSliceMapper.SetSliceFacesCamera(True)
    imageSliceMapper.StreamingOn()
    imageSlice = vtk.vtkImageSlice()
    imageSlice.SetMapper(imageSliceMapper)
    renderer.AddActor(imageSlice)

    setupCamera(renderer, imageSlice)

    renderWindow.Render()
    renderWindowInteractor.Start()

