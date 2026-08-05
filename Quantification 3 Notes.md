# Quantification 3 Notes

`Quantification 3.py` converts the saved prediction outputs into a formatted
Excel workbook containing scan-, class-, region-, pore-, interface-, spatial-,
texture-, and quality-control measurements.

## Installation

Install the project dependencies before running the script:

```powershell
python -m pip install -r requirements.txt
```

The additional quantification dependencies are SciPy, scikit-image, and
openpyxl. The script does not require a GPU.

## Basic usage

The physical pixel dimensions must come from the acquisition metadata. The
defaults are 0.22 µm by 0.22 µm only because that is the calibration used by
the preceding quantification script.

```powershell
python "Quantification 3.py" `
  --prediction-dir prediction `
  --pixel-width-um 0.22 `
  --pixel-height-um 0.22
```

The default output is `prediction/Quantification 3.xlsx`. If that file already
exists, either select another path or explicitly replace it:

```powershell
python "Quantification 3.py" `
  --prediction-dir prediction `
  --output "prediction/Tissue measurements.xlsx" `
  --pixel-width-um 0.22 `
  --pixel-height-um 0.22 `
  --overwrite
```

## Automatically skipped filenames

The script intentionally excludes every file whose name ends in `20261.png` or
`20261.czi`. Matching is case-insensitive. A message is printed for every
excluded full segmentation or original image so the omission is visible in the
processing log.

## Including original-image color and texture

The prediction outputs alone support geometry, topology, interfaces, spatial
distances, confidence, and uncertainty. Supply the directory containing the
matching raster or CZI originals to also calculate RGB, optical-density,
Hematoxylin/Eosin, gradient, GLCM, local-entropy, structure-tensor, and Hessian
features:

```powershell
python "Quantification 3.py" `
  --prediction-dir prediction `
  --original-dir "Original Scans" `
  --pixel-width-um 0.22 `
  --pixel-height-um 0.22
```

Raster originals are matched by filename stem. CZI outputs ending in
`__scene_###` are matched to the corresponding CZI stem and scene. The original
scene and segmentation must have identical native dimensions. If a unique
match cannot be found, only that scan's color/texture stage is skipped and the
reason is recorded on the `QC` sheet.

## Large whole-slide images

Hard pixel counts, hard areas, probability-weighted areas, confidence,
entropy, and probability margins are always calculated at native output
resolution. Connected components, holes, skeletons, distance transforms,
interfaces, curvature, and texture can require many working arrays and are
controlled independently by `--analysis-downsample`.

Native-resolution advanced analysis gives the most detail:

```powershell
python "Quantification 3.py" --prediction-dir prediction --analysis-downsample 1
```

For very large slides, begin with 4× analysis downsampling:

```powershell
python "Quantification 3.py" `
  --prediction-dir prediction `
  --original-dir "Original Scans" `
  --analysis-downsample 4 `
  --chunk-rows 128
```

The physical units are corrected for the downsampling factor, but very small
regions, pores, narrow interfaces, and fine texture can disappear. Use one
downsampling value consistently across specimens. If memory remains limited,
increase the factor to 8. If curvature or skeleton measurements are not needed,
`--skip-curvature` and `--skip-skeleton` reduce processing time and memory.

The detailed `Region Details` and `Pore Details` sheets are capped at 500,000
rows by default because Excel worksheets have a hard row limit. The retained
objects are the largest objects encountered within each class. Class-level
counts and distribution summaries still use every detected object. Change the
caps with `--max-region-rows` and `--max-pore-rows`.

## Workbook structure

- `README`: input paths, thresholds, calibration, and workbook navigation.
- `Scan Summary`: one row per scan with dimensions, total/tissue area,
  uncertainty, boundary-map results, and missing-class checks.
- `Class Summary`: one row per scan and class with exact area, expected area,
  region distributions, morphology, topology, skeleton, thickness, spatial,
  and confidence measurements.
- `Region Details`: retained connected-component measurements and confidence.
- `Pore Details`: retained closed-hole measurements.
- `Interfaces`: every unordered class pair, including adjacency, contact length,
  curvature, roughness, uncertainty, and Bone/Fibrocartilage transition width.
- `Spatial Distances`: every directed source-to-target class comparison.
- `Distance Bands`: class area by distance from Bone.
- `Texture Color`: optional original-image stain, color, and texture features.
- `QC`: missing inputs, all-background scans, unavailable measurements, and
  interpretation warnings.
- `Metric Definitions`: concise definitions and limitations.

All measurements are written as typed numeric Excel cells. Fractions are stored
as numbers between 0 and 1 and displayed as percentages. Identifiers, class
names, paths, and explanatory status fields are text.

## Measurement definitions

### Areas and probabilities

- **Hard area:** number of pixels whose final class ID equals the class,
  multiplied by calibrated pixel area.
- **Hard image-area fraction:** hard class pixels divided by all image pixels.
- **Hard reference-area fraction:** tissue-class pixels divided by all
  non-Background pixels. Background is divided by total image area.
- **Expected area:** sum of the class probabilities multiplied by pixel area.
  The float32 `Probabilities/*.npy` volume is preferred. If it is absent, the
  six 8-bit `Grayscale` probability maps are used.
- **Confidence-weighted area:** the expected area. It is reported using the
  clearer `Expected area` name.
- **Normalized entropy:** Shannon entropy divided by `log(6)`, so zero is
  decisive and one is maximally uncertain.
- **Top-two margin:** difference between the largest and second-largest class
  probabilities. A low margin indicates ambiguity.
- **Bone/Fibrocartilage ambiguity:** pixels for which Bone and Fibrocartilage
  are the two most likely classes and the probability margin is below the
  configured threshold.

### Connected tissue regions

Each region is an 8-connected component of one hard class mask.

- **Region count and density:** number of components and components per square
  millimeter of image.
- **Largest-region area fraction:** largest component divided by all class area.
- **Fragmentation fraction:** one minus the largest-region area fraction.
- **Perimeter:** physical contour length from marching-square contours.
- **Circularity:** `4π × area / perimeter²`.
- **Solidity:** region area divided by convex-hull area.
- **Convexity:** convex-hull perimeter divided by region perimeter.
- **Boundary roughness:** region perimeter divided by convex-hull perimeter.
- **Aspect ratio:** major-axis length divided by minor-axis length.
- **Elongation:** one minus minor-axis length divided by major-axis length.
- **Orientation:** calibrated physical major-axis direction relative to the
  positive image X axis. Major/minor axes and eccentricity are calculated from
  physical coordinates, so unequal pixel width and height are respected.
- **Orientation regularity:** length of the area-weighted axial orientation
  vector. Values near one indicate alignment.
- **Clark-Evans index:** observed mean centroid nearest-neighbor distance divided
  by the Poisson expectation. Values below one suggest clustering and values
  above one suggest dispersion. No border correction is applied.

Regions touching the outer image border are flagged because their shape and
size may be truncated by the field of view.

### Closed holes and porosity

A closed hole is an 8-connected non-class island enclosed after binary hole
filling of one class mask. The script reports hole count, area and diameter
distributions, density, nearest-neighbor spacing, enclosed porosity, and Euler
number.

These holes are not automatically biological pores or transparent cells. A
hole can contain another predicted tissue, an imaging void, or a segmentation
error. Biological interpretation should be validated against the RGB image.

### Skeleton, local thickness, and fracture proxy

- **Skeleton length:** physical length of the 8-connected medial skeleton.
- **Endpoints:** skeleton pixels with one neighbor.
- **Junctions:** connected clusters of skeleton pixels with at least three
  neighbors.
- **Local thickness:** twice the distance to the nearest class boundary at
  skeleton pixels.
- **Internal-void skeleton density:** skeleton length of enclosed holes divided
  by tissue area. This is a fracture/crack proxy, not a validated crack
  classifier.

### Interfaces and boundaries

- **Interface length:** physical length of horizontal and vertical pixel edges
  shared by two different hard classes.
- **Contact fraction:** pair interface length divided by all interclass contact
  length for that class.
- **Interface roughness proxy:** interface length divided by the diagonal of
  the interface's overall bounding box. Multiple separated contacts can raise
  this value.
- **Interface curvature:** mean absolute divergence of the signed-distance unit
  normal sampled on both sides of the interface.
- **Uncertain interface fraction:** interface pixels with a top-two probability
  margin below the configured threshold.
- **Bone/Fibrocartilage transition-width proxy:** ambiguous Bone/Fibrocartilage
  area divided by their hard-interface length. It is undefined if the hard
  masks do not touch.

### Spatial organization

- **Pixel-to-target distance:** Euclidean distance from source-class pixel
  centers to the nearest target-class pixel.
- **Region-centroid distance:** nearest target-region centroid distance for each
  source-region centroid.
- **Proximity fraction:** fraction of source area within `--proximity-um` of the
  target. The Muscle-to-Fibrocartilage row provides the requested conditional
  spatial relationship without falsely claiming overlap between exclusive hard
  masks.
- **Distance bands:** each class's area distribution in fixed-width bands from
  Bone.
- **Radial position and direction:** region-centroid distance and direction from
  the center of the segmentation field.

Because the final hard classes are mutually exclusive, literal Muscle area
inside the hard Fibrocartilage mask is always zero. Proximity and probability
ambiguity are more meaningful measurements for that relationship.

### Original-image color and texture

- RGB channel means/SDs and combined RGB variation.
- Per-channel optical density using `-log((channel + 1) / 256)`.
- H/E/residual-DAB values from scikit-image's standard RGB-to-HED stain matrix.
- Grayscale mean, SD, and Shannon entropy.
- Sobel gradient magnitude.
- Local entropy in a disk controlled by `--texture-entropy-radius`.
- Masked symmetric GLCM contrast, homogeneity, energy, and correlation using
  one-pixel offsets at 0°, 45°, 90°, and 135°.
- Structure-tensor coherence, dominant orientation, and axial regularity.
- Hessian ridge strength from the negative minimum eigenvalue at sigma 1.

These values depend on acquisition settings, staining, white balance,
compression, and analysis downsampling. Compare specimens only when those
conditions are controlled. The default H/E stain vectors should be validated
for the laboratory protocol before treating stain values as concentrations.

## Important limitations

1. Accurate physical units require correct pixel width and height for every
   acquisition and scene.
2. Boundary, skeleton, pore, and topology measurements are more sensitive to
   segmentation noise than area measurements.
3. Do not apply unvalidated erosion, dilation, opening, closing, or small-object
   removal solely to improve the appearance of the metrics; each operation can
   bias tissue area and topology.
4. The saved outputs do not preserve individual overlapping-tile predictions.
   Therefore tile-to-tile disagreement cannot be calculated after prediction.
5. This is two-dimensional analysis. It cannot measure three-dimensional
   volume, surface area, pore connectivity, or tissue orientation through depth
   without registered serial sections or volumetric imaging.
6. These measurements are computational descriptors, not diagnoses. Validate
   them against expert annotations and specimen-level biological endpoints.

## Useful options

```text
--analysis-downsample N       Heavy-metric scale; 1 is native resolution.
--chunk-rows N                Rows read per native probability pass.
--low-confidence-threshold X  Hard-region low-confidence cutoff.
--uncertainty-margin X        Maximum top-two margin considered ambiguous.
--high-entropy-threshold X    Normalized-entropy cutoff.
--proximity-um X              Source-to-target proximity distance.
--distance-bin-um X           Width of distance bands from Bone.
--texture-levels N            Quantization levels for GLCM texture.
--texture-entropy-radius N    Local-entropy disk radius at analysis scale.
--max-region-rows N           Workbook region-detail cap.
--max-pore-rows N             Workbook pore-detail cap.
--skip-curvature              Skip signed-distance curvature calculations.
--skip-skeleton               Skip skeleton and thickness calculations.
--overwrite                   Permit replacing an existing workbook.
```
