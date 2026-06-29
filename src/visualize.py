import itk
import vtk
import os
import numpy as np
from vtk.util import numpy_support


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

def _itk_to_uint8_array(image, vmax_percentile=99.5):
    """Window an itk float image to 0-255 for color blending / volume rendering."""
    arr = itk.GetArrayFromImage(image)
    vmax = np.percentile(arr[arr > 0], vmax_percentile)
    arr8 = np.clip(arr, 0, vmax) / vmax * 255.0
    return arr8.astype(np.uint8)


def _blend_overlay(gray_u8, mask1, mask2, color1=(40, 200, 40), color2=(230, 90, 0),
                    alpha=0.45, alpha_overlap=0.55):
    """Blend a grayscale volume with two colored, semi-transparent mask overlays.

    Uses the same color convention as render_3d_comparison: green = scan 1
    only, orange = scan 2 (registered) only, and a blended color where both
    masks agree (spatial overlap).
    """
    rgb = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float32)
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    overlap = mask1 & mask2

    c1 = np.array(color1, dtype=np.float32)
    c2 = np.array(color2, dtype=np.float32)
    c_overlap = (c1 + c2) / 2

    rgb[only1] = (1 - alpha) * rgb[only1] + alpha * c1
    rgb[only2] = (1 - alpha) * rgb[only2] + alpha * c2
    rgb[overlap] = (1 - alpha_overlap) * rgb[overlap] + alpha_overlap * c_overlap

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _numpy_rgb_to_vtk_image(rgb_arr, spacing=(1, 1, 1)):
    """rgb_arr: (z, y, x, 3) uint8 -> vtkImageData with 3-component scalars."""
    flat = rgb_arr.reshape(-1, 3)
    vtk_data = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_data.SetNumberOfComponents(3)
    img = vtk.vtkImageData()
    img.SetDimensions(rgb_arr.shape[2], rgb_arr.shape[1], rgb_arr.shape[0])  # x, y, z
    img.SetSpacing(*spacing)
    img.GetPointData().SetScalars(vtk_data)
    return img


def interactive_slice_viewer(image1, mask1_array, mask2_array, screenshot_path,
                              interactive=True, initial_slice=None):
    """Scrollable 2D slice-by-slice view of scan 1, with both tumor masks
    color-highlighted directly on each slice (green=scan1, orange=scan2
    registered, blended where they overlap).

    This generalizes the static `plot_2d_comparison` figure (which only
    shows one chosen slice) into something explorable: scroll the mouse
    wheel to move through the full stack and watch the tumor outline
    appear/disappear/change shape slice by slice - a 2D way of building up
    a 3D mental picture without needing a full surface reconstruction.

    Built the same way as the `render()` example function we were given:
    itk image -> vtkImageData -> vtkImageSliceMapper/vtkImageSlice ->
    vtkInteractorStyleImage in "image slicing" mode (mouse wheel = change
    slice). The only difference is that here we feed it a pre-blended RGB
    volume instead of the raw grayscale one, so the colored overlay moves
    together with the slice.
    """
    gray_u8 = _itk_to_uint8_array(image1)
    rgb = _blend_overlay(gray_u8, mask1_array.astype(bool), mask2_array.astype(bool))
    spacing = tuple(image1.GetSpacing())
    vtk_image = _numpy_rgb_to_vtk_image(rgb, spacing)

    if initial_slice is None:
        combined = mask1_array | mask2_array
        initial_slice = int(np.argmax(combined.sum(axis=(1, 2))))

    mapper = vtk.vtkImageSliceMapper()
    mapper.SetInputData(vtk_image)
    mapper.SetOrientation(2)  # slice along the same axis used everywhere else (numpy axis 0)
    mapper.SetSliceNumber(initial_slice)

    image_slice = vtk.vtkImageSlice()
    image_slice.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(image_slice)
    renderer.SetBackground(0.05, 0.05, 0.05)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(800, 800)

    image_style = vtk.vtkInteractorStyleImage()
    image_style.SetInteractionModeToImageSlicing()

    setupCamera(renderer, image_slice)

    has_display = bool(os.environ.get("DISPLAY"))
    opened_onscreen = False
    if interactive and not has_display:
        print("  (No DISPLAY detected - skipping interactive slice viewer, screenshot only)")
    if interactive and has_display:
        try:
            ren_win.Render()
            opened_onscreen = True
        except Exception:
            opened_onscreen = False

    if not opened_onscreen:
        ren_win.SetOffScreenRendering(1)
        ren_win.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInputBufferTypeToRGB()
    w2if.SetInput(ren_win)
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(screenshot_path)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    print(f"  Saved slice-viewer screenshot to {screenshot_path} (slice {initial_slice})")

    if interactive and opened_onscreen:
        try:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(ren_win)
            interactor.SetInteractorStyle(image_style)
            interactor.Initialize()
            print("  Opening slice viewer - scroll to change slice, drag to adjust "
                  "window/level, close window to continue...")
            interactor.Start()
        except Exception as e:
            print(f"  (Interactive window unavailable: {e})")

