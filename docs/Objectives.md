# Objectives

These are the targets I am setting before collecting data or training the model. I will enter the actual results only after running the experiments. Missing results will stay marked as pending.

## MO-1: Measurement accuracy

**What I will measure:** The absolute error, in millimetres, between the card's actual width and height and the values reported by the system.

**How I will measure it:** I will use the 10 photographs reserved for testing. I will measure the card with a calliper and run each original photograph through the completed pipeline. For each accepted photograph, I will calculate the absolute width error and height error, then average those two errors. I will report the mean, median and largest error across the accepted photographs. I will also report percentage error and show every photograph in the results table.

**Target:** Mean absolute dimension error of at most 2.0 mm, with at least 9 of the 10 test photographs accepted. I chose 2.0 mm because it is a useful first target for measuring a card with a phone, while still requiring more than a rough visual estimate.

**Conditions:** The same calibrated phone settings, a rigid card, a fully visible measured ArUco marker in the same plane, and normal indoor lighting.

**Not included:** Bent objects, hidden edges or a marker placed on a different plane. If the system refuses a photograph, I will count and explain the refusal rather than treating it as a correct measurement.

**Result:** Pending. Evidence will go in `measurement/results/per_instance.csv` and `measurement/results/summary.json`.

## MO-2: Inference time

**What I will measure:** The time needed to process a raw photograph and produce a measurement and overlay, once the model is loaded.

**How I will measure it:** I will run the complete inference process five times on the same photograph after two warmup runs. I will report all five timings, their median, the smallest and largest time, and the difference between the two.

**Target:** Median processing time no greater than 1500 ms on the machine used for the assessment.

**Conditions:** The same image and machine for every run, with a 320 by 320 model input. Timing includes undistortion, segmentation, marker detection, measurement and overlay creation. Model loading and taking the photograph are excluded.

**Not included:** Time spent installing packages, starting Python or manually taking the photograph.

**Result:** Pending. Evidence will go in `measurement/results/benchmark.json`.

## Other results I need to report

**Camera calibration:** Number of successful checkerboard images, the camera matrix, distortion coefficients and reprojection error. I will keep the original calibration images and saved parameters.

**Segmentation:** Mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall on the untouched test split. I will include the training history and the individual test results. I will choose the checkpoint using validation results only.

**Raw versus undistorted images:** Measurement errors for the same photographs in both conditions, including the number of accepted and refused photographs. I will compare errors on matched accepted photographs and state the limits of that comparison.

**Failure case:** I will move the card from the centre toward the edge of the frame while keeping the marker visible. I will report the measurement error and refusal count at each position.

**Refusals:** I will record the thresholds, reasons for refusal and the share of photographs the system refuses to measure. A refusal is not a successful measurement.

**Hardware:** Before reporting performance, I will record the CPU, GPU, RAM, operating system, phone model and camera resolution. I will include the exact software versions and random seed used for training.
