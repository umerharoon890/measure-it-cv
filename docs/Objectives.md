# Objectives

I set these targets before collecting data and training the model. I have kept the original targets here and added the measured results, including failures and changes made during testing.

## MO-1: Measurement accuracy

**What I measure:** The absolute error, in millimetres, between the card's actual width and height and the values reported by the system.

**Method:** I use 10 new photographs reserved for physical measurement testing. The card's measured dimensions are 57 mm wide and 87 mm tall. For each accepted photograph, I calculate the absolute width and height errors and average them to obtain one error per photograph. I report the mean, median, worst error and percentage error. Refused photographs count against the acceptance requirement and are not included in the error average.

**Original target:** Mean absolute dimension error of at most 2.0 mm, with at least 9 of the 10 photographs accepted. I chose 2.0 mm as a useful starting target for measuring a card with a phone rather than settling for a rough visual estimate.

**Conditions:** The same calibrated phone, a rigid card, a fully visible ArUco marker with a measured 40 mm black square, the card and marker on the same plane, and normal indoor lighting. The ground-truth dimensions were checked manually; their precision is limited by the measuring tool. These results do not establish accuracy for other objects or nonplanar scenes.

**Result before the boundary fix:** All 10 photographs were accepted. Mean error was 2.015 mm, median error 2.276 mm, worst error 4.302 mm and mean percentage error 2.929%. The original accuracy target was narrowly missed.

**Final result after the boundary fix:** Nine photographs were accepted and one was refused because its mask fell within the 15-pixel boundary safety margin. Mean error across the nine accepted photographs was **1.964 mm**, median error **2.228 mm**, worst error **4.302 mm**, and mean percentage error **2.852%**. The original target was met with exactly the minimum required 9 of 10 accepted photographs.

The lower final mean error does not mean the network's measurements improved. The new refusal rule removed one photograph from the error calculation. I preserve both results so this distinction is visible.

**Evidence:** `measurement/results/per_instance.csv`, `measurement/results/summary.json`, `measurement/results/per_instance_before_edge_fix.csv` and `measurement/results/summary_before_edge_fix.json`.

## MO-2: Inference time

**What I measure:** Processing time for one raw photograph after the model has been loaded. The timed path includes image preparation and undistortion, segmentation, ArUco detection, dimension estimation and overlay generation. It excludes model loading, Python startup and taking the photograph.

**Method:** Two warmup runs followed by five timed runs on the same photograph, reporting each timing, the median, minimum, maximum and spread.

**Original target:** Median processing time no greater than 1,500 ms on the assessment machine.

**Implementation detail:** The original plan specified a 320 by 320 model input. The trained model and final pipeline actually use **288 by 384 pixels**, with the processed image at **756 by 1008 pixels**. I kept the timing target unchanged.

**Results:**

| Run | Processing time |
| --- | ---: |
| 1 | 452.26 ms |
| 2 | 433.68 ms |
| 3 | 453.57 ms |
| 4 | 419.12 ms |
| 5 | 703.83 ms |

The **median was 452.26 ms**. The minimum was 419.12 ms, the maximum was 703.83 ms, and the spread was 284.71 ms. All five runs produced a measurement. **MO-2 was met.**

**Evidence:** `measurement/results/benchmark.json`.

## Segmentation and calibration

I collected 35 checkerboard photographs for calibration. The saved camera parameters and calibration script are in `calibration/`; the reported reprojection RMS was approximately **2.363 px**. This is not a guarantee that each edge of the image is perfectly corrected.

I captured 89 original card photographs and manually labelled their processed masks. The final split was **60 training, 19 validation and 10 test photographs**. I used validation results to select the U-Net checkpoint at epoch 20, not the test data.

On the untouched segmentation test split, mean mask IoU was **0.9688**, pixel precision **0.9777**, and pixel recall **0.9919**. The reported mask mAP@0.5 and mAP@0.5:0.95 were both **1.0000** using the evaluation's one-proposal-per-image definition. Those AP results should not be interpreted as performance on a general multi-object detection benchmark.

**Evidence:** `calibration/parameters.npz`, `models/training_config.json`, `models/evaluation/test_metrics.json`, and the saved dataset split lists.

## Raw versus undistorted photographs

I ran the same 10 physical measurement photographs with the final pipeline in raw and undistorted modes. Nine were accepted in both conditions and one was refused in both because the object was too close to the image boundary.

Across the nine matched accepted photographs, the raw-image mean error was **3.272 mm** and the undistorted mean error was **1.964 mm**. This was a **1.308 mm reduction, about 40.0%**. The raw and undistorted median errors were 2.840 mm and 2.228 mm, respectively. The worst errors were 5.875 mm and 4.302 mm.

Undistortion did not help every photograph: two of the nine matched cases had higher error after correction. Also, the segmentation network was trained on undistorted images, so the comparison describes the complete pipeline rather than the isolated geometric effect of lens correction.

**Evidence:** `measurement/results/undistortion_comparison.csv` and `measurement/results/undistortion_summary.json`. Earlier comparison files are preserved with the `_before_edge_fix` suffix.

## Stress test and refusal behavior

I moved the card from the centre towards the left edge of the camera frame while keeping the marker visible and in the same plane. The results for the fully visible card were:

| Photograph | Predicted width | Predicted height | Mean error | Result |
| --- | ---: | ---: | ---: | --- |
| `stress_01.jpg` | 59.40 mm | 88.51 mm | 1.96 mm | Measured |
| `stress_02.jpg` | 58.85 mm | 88.41 mm | 1.63 mm | Measured |
| `stress_03.jpg` | 59.85 mm | 88.97 mm | 2.41 mm | Measured |

In `stress_04.jpg` the card was partly outside the frame. Initially the pipeline accepted it as **48.09 by 89.24 mm**, with **0.9463** mask confidence. Its width was wrong by 8.91 mm, yet the original exact-border check did not flag the incomplete object. This showed that high mask confidence alone could not establish trustworthy measurement geometry.

I changed the boundary check to refuse a mask entering a **15-pixel margin** around the processed image. After the change, `stress_03.jpg` was still measured, and `stress_04.jpg` was refused with the reason `Object too close to image boundary`. The new rule can reject some fully visible objects; that is the deliberate trade-off for avoiding measurements near an uncertain image boundary. It cannot catch every kind of occlusion or segmentation error.

The pipeline also refused a photograph with no marker (`Reference marker not found`) and one with no card (`Object not found`). These are demonstrated cases, not a claim that every failure scenario is detected.

**Evidence:** `measurement/stress_images/`, `inference/outputs/stress_01_overlay.json` through `stress_04_overlay.json`, and the missing-reference and missing-object test outputs. The initial `stress_04` acceptance is recorded in the experiment notes and terminal results; the output JSON was regenerated after the fix.

## Hardware and reproducibility

The photographs were taken with a Samsung Galaxy S10+ at a calibration image resolution of **3024 by 4032 pixels**. Training and inference used PyTorch **2.5.1+cu118** and OpenCV **4.10.0** with a GTX 1050 GPU. The training seed was **2026**. The final README will record the full CPU, RAM, operating system, dependency and run-command details.
