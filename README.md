# vtk_itk : A Longitudinal Study of Tumor Progression

This project performs a longitudinal study of a tumor from two scans
of the same patient acquired at different dates.

## 1. Installation and Usage

Run from the root of the repository.

```bash
uv run main.py [arguments]
```

### Arguments

- `--display {none,slice,comparison,volume}` *(default: `comparison`)*
  - `none`: no interactive window. It only saves the 2D comparison figure to `output/`.
  - `slice`: scrollable 2D slice viewer (mouse wheel) with both tumor contours overlaid.
  - `comparison`: interactive 3D view: tumor at timepoint 1 in **green**, timepoint 2 in **orange**.
  - `volume` : 3D volume ray-casting of the brain with the segmented tumor surface.
- `--dont-plot` : disables generation of the 2D Matplotlib figure.

All outputs of the program are written to `output/`.

## 2. Technical choices

### 2.1 Recalage

We estimate the alignment on the original images and apply the result to the mask.
The two acquisitions share the same 1 mm-isotropic grid and differ mainly in their
physical `origin`, so the misalignment to recover is essentially a repositioning.

- **Initialization** : A `CenteredTransformInitializer` aligns the centers of mass of the two volumes, giving the optimizer a good starting point.
- **Stage 1 : Rigid** (`VersorRigid3DTransform`) : It captures the dominant difference between the two scan.
- **Stage 2 : Affine** (`AffineTransform`) : Initialized from the rigid result, it absorbs residual global scaling/shear.

**Metric : Mattes Mutual Information.** : 

Even though both scans share the same modality, MI is robust to the intensity drift that occurs between
separate acquisitions, and it is a standard, well-behaved choice for 3D MRI registration.

**Applying the transform** : 

The binary mask of scan 2 is resampled onto scan 1's using nearest-neighbor interpolation, so the mask stays strictly binary.

**Parameters** :

| Stage | Transform | Levels | Shrink | Smoothing | Learning rate | Min. step | Iterations |
|-------|-----------|--------|--------|-------------|---------------|-----------|------------|
| Rigid | VersorRigid3D | 3 | `[4,2,1]` | `[2,1,0]` | 1.0 | 0.001 | 100 |
| Affine | Affine | 2 | `[2,1]` | `[1,0]` | 0.5 | 0.0005 | 150 |

Optimizer: `RegularStepGradientDescentOptimizerv4`

### 2.2 Segmentation

**Method: `ConfidenceConnected` region growing (semi-automatic).**

**Seeds are hardcoded** They were chosen **empirically**: we tried
several candidate seed positions in each scan and kept the ones that produced the
best-looking segmentation.

Two different seeds are used because the tumor's position differs between the two
scan. The **same method and parameters** are applied to both scans so the two segmentations are directly comparable.

- Seeds : scan 1 = `[103, 77, 51]`, scan 2 = `[91, 89, 53]`
- Parameters : `multiplier=1.2`, `number_of_iterations=2`, `initial_neighborhood_radius=2`

**Known limits:** It is sensitive to seed placement and to `multiplier` and it requires one seed per scan.

### 2.3 Change analysis

All metrics are computed on the two masks **in the same physical space** (after
registration):

- **Volume**
- **Absolute / Relative change**
- **Dice overlap** : `2·|A ∩ B| / (|A| + |B|)`

### 2.4 Visualization

Four complementary views are available:

- **2D comparison figure** : Matplotlib: scan 1 contour (green), scan 2 (orange), and a change map (blue = regressed, red = new growth).
- **Slice viewer** (`--display slice`) : It will display a 2D slice of the brain and the tumor. *(VTK `vtkImageSliceMapper`)*
- **3D comparison** (`--display comparison`) : It will display a 3D viewer with the tumor at the first timestamp in green, and the second timestamp in orange to study the change. *(Surfaces via Marching Cubes)*
- **Volume ray-casting** (`--display volume`) : It allows a viewing of the tumor with the skull a bit visible, in 3D as well. *(`vtkSmartVolumeMapper`)*

## 3. Results

### Registration

Both stages converges well within their number of iterations
and the affine stage measurably improves the Mattes MI over the rigid one :

| Stage | Iterations to converge | Final Mattes MI |
|-------|------------------------|-----------------|
| Rigid | 9 | −0.7100 |
| Affine | 12 | −0.7546 |

The rigid stage already reaches a good alignment quickly and the affine refinement
lowers the metric further from −0.7100 to −0.7546, confirming the two-stage approach
was worthwhile.

### Tumor change

```
--- Tumor change analysis ---
Volume scan 1             : 6.05 cm3
Volume scan 2 (registered): 3.88 cm3
Absolute change           : -2.17 cm3
Relative change           : -35.9 %
Dice overlap (spatial)    : 0.588
```

**Interpretation.** Between the two timepoints the tumor **regressed**, shrinking from
6.05 cm³ to 3.88 cm³, a reduction of 2.17 cm³ (−35.9 %).

## 4. Difficulties encountered

- **Segmentation tuning** : finding a `multiplier` that fills correctly the tumor
  and finding the seed placement had to be chosen after several trials.
- **Automatic segmentation** : an automatic pipeline was attempted but dropped for
  time.

## 5. Limitations

- **Automatic segmentation** : the current approach is semi-automatic (with hardcoded
  seeds). A fully automatic method could be interesting to have.
- **Registration validation** : a quantitative or visual check
  would confirm alignment quality and allow for an effective comparison of the various registration algorithms.
- **Voxel-intensity change** : A voxel intensity difference between the two registered scans could be an interesting metric.

## 6. Project structure

```
.
├── main.py                 # entry point
├── data/                   # case6_gre1.nrrd, case6_gre2.nrrd
├── output/                 # generated transform, figures and screenshots
└── src/
    ├── registration.py     
    ├── segmentation.py     
    ├── analysis.py         # Compute metrics
    └── visualization.py
```

## 7. Contributors

- **alexis.meunier** : project setup, segmentation, final touches
- **anis.feore** : recalage
- **lucil.finkelstein** : visualization tools
- **johan.emmanuelli** : visualization tools

## Note on version control

The project was managed on GitLab, hence the `.gitlab-ci.yml` at the root. The
issues, merge requests and milestones we created therefore live on GitLab and are not
visible from a GitHub mirror.