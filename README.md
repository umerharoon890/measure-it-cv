# Measure It

I built this project to estimate the physical width and height of a playing card from a photograph taken with a calibrated phone camera. The system uses a segmentation model to find the card and a printed ArUco marker to convert image coordinates into millimetres. It also refuses to report a measurement when some essential information is missing or the object is too close to the image boundary.

The project was developed for the xis.ai Measure It technical assessment. The code, calibration photographs, original object images, masks, training results, model weights and experimental evidence are included in this repository.

## Example measurement

From the repository root, with the Python environment activated:

```powershell
python inference/measure.py --image "inference/new_images/live_test_01.jpg"
```

The command prints a status, width, height and mask confidence, then saves a visual overlay and a JSON record in `inference/outputs/`. For this previously unseen photograph, the system reported 59.27 mm by 88.65 mm, compared with the card's measured dimensions of 57 mm by 87 mm.

For a photograph without the reference marker, it refuses with `Reference marker not found`. For a photograph without the card, it refuses with `Object not found`.

## How the measurement works

1. Read a full-resolution photograph from the calibrated Samsung Galaxy S10+ camera. Rotate landscape photographs to the calibrated portrait orientation and check the image resolution.
2. Use the saved camera matrix and distortion coefficients to undistort the image, then resize it to 756 by 1008 pixels for inference.
3. Run a small U-Net on a 288 by 384 pixel copy to predict the playing-card mask. Retain its largest connected component.
4. Detect a 4x4 ArUco marker with ID 0. Its printed square is 40 mm wide.
5. Use the four marker corners to estimate a planar perspective transformation in millimetres. Transform the segmented card contour through that mapping and fit a rotated rectangle to estimate its shorter and longer sides.
6. Report width, height, mask confidence, an overlay and a JSON result, or refuse the measurement when a required check fails.

The marker and card must lie flat on the same plane. This is a single-object, planar measurement method, not a general 3D measurement system. The mask confidence describes the segmentation prediction; it is not a guarantee of dimensional accuracy.

The implementation is in `inference/measure.py`. The `prepare_image`, `segment_card`, `find_reference`, `measure_in_mm` and `main` functions cover the main stages.

## Hardware and software

| Item | Used for this project |
| --- | --- |
| Camera | Samsung Galaxy S10+, full-resolution 3024 by 4032 photographs |
| Calibration target | Checkerboard, 9 by 7 inner corners, 18 mm square size |
| Scale reference | Printed ArUco DICT_4X4_50 marker, ID 0, 40 mm square |
| Measured object | Joker playing card, 57 mm wide and 87 mm tall |
| CPU | Intel Core i7-7700HQ, 2.80 GHz |
| GPU | NVIDIA GeForce GTX 1050; Intel HD Graphics 630 also present |
| RAM | 15.9 GB |
| Operating system | Windows 10 Home |
| Python | 3.11.4 |
| PyTorch | 2.5.1+cu118 |
| OpenCV | 4.10.0 |
| NumPy | 1.26.4 |

## Setup

Commands below use PowerShell. Run them from the repository root.

```powershell
git clone -b dev-test01 https://github.com/umerharoon890/measure-it-cv.git
cd measure-it-cv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 opencv-contrib-python==4.10.0.84
python -m pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

The OpenCV contrib package is required for `cv2.aruco`. The listed PyTorch build supports the CUDA 11.8 environment used during development. For CPU-only systems, use an appropriate CPU PyTorch installation instead; inference will run on CPU but may be slower. These are the core packages directly imported by the project scripts.

The trained weights and calibration parameters are already included, so retraining and recalibration are **not** needed to try an example measurement.

## Run the pipeline

```powershell
python inference/measure.py --image "inference/new_images/live_test_01.jpg"
```

To run a photograph without applying distortion correction:

```powershell
python inference/measure.py --image "inference/new_images/live_test_01.jpg" --skip-undistortion
```

The input must come from the same calibrated camera, with the same image geometry and an original resolution matching 3024 by 4032 after orientation handling. The program saves an overlay JPEG and JSON record in `inference/outputs/`. The `--output` option can change the overlay location.

The system can refuse a measurement if the card is missing, the predicted mask is too small, mean foreground mask confidence is below 0.85, the card mask reaches the 15-pixel image-edge safety margin, the reference marker is absent or unusable, or dimensions cannot be estimated. These checks reduce some unreliable results but cannot detect every possible failure.

## Dataset and training

I photographed and manually annotated 89 images of the same playing card and ArUco marker. The raw photographs are in `dataset/raw/`. The prepared, undistorted 756 by 1008 images are in `dataset/images/`, and the manually drawn binary masks are in `dataset/masks/`. `dataset/image_manifest.csv` relates processed names to the original camera files.

The original plan called for 100 images. The final dataset was smaller, and missing filenames were removed from the preassigned split lists before training. Existing assignments were preserved:

| Split | Images |
| --- | ---: |
| Training | 60 |
| Validation | 19 |
| Test | 10 |

The split lists are in `dataset/splits/`. The validation set was used for checkpoint selection; the test split was not used to tune the model. The model is a 1,944,049-parameter U-Net trained with BCE plus Dice loss, AdamW, a batch size of 2, learning rate 0.0003 and random seed 2026. Training stopped after epoch 28; the selected checkpoint is from epoch 20 (validation mask IoU 0.9700).

To validate the stored masks, train again, or evaluate the saved checkpoint:

```powershell
python dataset/validate_masks.py
python models/train.py
python models/evaluate.py
```

Training overwrites `models/best_unet.pth` and the saved training records. If you only want to reproduce inference or evaluation with the submitted checkpoint, **do not run the training command**. The calibration and data preparation scripts are `calibration/calibrate.py` and `dataset/prepare_images.py`; running them also regenerates their respective saved outputs.

### Segmentation test results

| Metric | Held-out segmentation test result |
| --- | ---: |
| Mean mask IoU | 0.9688 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 1.0000 |
| Pixel precision | 0.9777 |
| Pixel recall | 0.9919 |

There is one annotated card and one proposed prediction per image. AP is computed with 101-point interpolation from the single-object predictions; it should not be interpreted as a general multi-object detection benchmark. Detailed results are saved in `models/evaluation/test_metrics.json` and `models/evaluation/test_per_image.csv`.

## Physical measurement experiments

The dimensional accuracy experiment used **10 new photographs**, distinct from the segmentation dataset. These are 10 repeated measurements of **one physical playing card**, not 10 different objects. The reference dimensions were 57 by 87 mm. Each accepted photograph contributes the average of the absolute width and height errors. Refused photographs are counted separately and are not included in the error mean.

| Metric | Final result |
| --- | ---: |
| Accepted photographs | 9/10 |
| Refused photographs | 1/10 |
| Mean absolute dimension error, accepted photographs | 1.964 mm |
| Median absolute dimension error | 2.228 mm |
| Worst absolute dimension error | 4.302 mm |
| Mean percentage error | 2.852% |

The original accuracy target was mean absolute error no greater than 2.0 mm with at least nine photographs accepted. The final pipeline meets that target. Before introducing the image-edge safety margin, all 10 photographs were accepted and mean error was 2.015 mm. The improvement in reported mean comes from refusing one photograph, not from making the other measurements more accurate. Both versions of the results are retained under `measurement/results/`.

### Raw versus undistorted

The same 10 photographs were processed with and without undistortion using the final boundary rule. Nine were accepted under both conditions.

| Metric on the nine matched accepted photographs | Raw | Undistorted |
| --- | ---: | ---: |
| Mean absolute dimension error | 3.272 mm | 1.964 mm |
| Median absolute dimension error | 2.840 mm | 2.228 mm |
| Worst absolute dimension error | 5.875 mm | 4.302 mm |

Undistortion was associated with a 1.308 mm (about 40.0%) lower mean error on these matched photographs. The segmentation model was trained on corrected images, so this is a comparison of the complete processing pipeline, not an isolated estimate of the lens-correction effect. Two photographs had slightly higher errors after correction.

### Processing time

After two warmup passes, five processing runs on the GTX 1050 system took 452.26, 433.68, 453.57, 419.12 and 703.83 ms. The median was **452.26 ms**, below the original 1500 ms target. The measured spread was 284.71 ms. Model loading and manual photography were excluded from the reported timing.

### Boundary stress test

Four new photographs placed the card progressively closer to the left image edge. The first three were measured, with mean dimension errors of approximately 1.96, 1.63 and 2.41 mm. The fourth card was partially outside the photograph. Initially, the pipeline incorrectly reported 48.09 by 89.24 mm with mask confidence 0.9463.

The original code only rejected masks touching the outermost pixel row or column. The predicted mask stopped slightly short of the left edge, so the cropped card passed. I replaced that check with a **15-pixel safety margin** around the processed image. The fourth case is now refused, while the third remains measurable. This is a conservative heuristic and can reject a fully visible card close to the edge.

Reproduce the saved experiments from the repository root:

```powershell
python measurement/evaluate_accuracy.py
python measurement/compare_undistortion.py
python measurement/benchmark.py --image "inference/new_images/live_test_01.jpg"
```

The physical measurement photographs, stress photographs, summaries, overlays and before-fix evidence are stored under `measurement/` and `inference/outputs/`. Run times can change with machine load, GPU selection and software environment.

## Model weights and integrity

The trained model is committed directly to the repository as [`models/best_unet.pth`](models/best_unet.pth) (7,833,658 bytes).

SHA-256:

```text
0AB2C30778D4F2140965811C7AC80B6AA3B1736DAC6F53FD6825B2A8B472034F
```

Verify it on Windows:

```powershell
Get-FileHash .\models\best_unet.pth -Algorithm SHA256
```

The SHA-256 above applies to the submitted checkpoint. It will change if the model is retrained.

## Limitations

This model was trained on one specific card and should not be expected to segment arbitrary objects reliably. The card and reference must be on the same plane and the camera must match the saved calibration. Lens correction does not eliminate segmentation or reference-detection errors. The boundary refusal margin does not detect every cropped or occluded card, and mask confidence alone cannot tell whether a physical measurement is accurate. Measurements should not be used where precise metrology or safety-critical decisions are required.

The initial plan, actual changes, original targets and final completion checklist are documented in [`docs/Plan.md`](docs/Plan.md), [`docs/Objectives.md`](docs/Objectives.md) and [`docs/Definition_of_Done.md`](docs/Definition_of_Done.md).

## Walkthrough and pull request

The required walkthrough recording and the pull request from `dev-test01` into `main` will be linked here after they are created.
