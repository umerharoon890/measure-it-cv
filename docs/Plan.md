# Plan

My aim is to measure the width and height of a real object from a photograph, then check the result with a calliper. I want the full process to work on a new photograph, not just on the images used to train the model.

## Object and camera

I plan to use a flat plastic card because it has clear edges and is easy to reposition. I will measure its actual dimensions before testing. I will use one phone camera and keep its focus, zoom, resolution and orientation unchanged throughout the project.

I will print a checkerboard with 9 by 7 inner corners, mount it flat and measure the printed square size. I need at least 20 usable photographs for OpenCV calibration. For scale, I will use a printed ArUco marker with ID 0. I will measure its outer square and keep it flat beside the card. The card and marker must be in the same plane for the conversion to millimetres to be valid.

## Work plan (8 active hours)

1. Prepare repository and printed targets: 30 minutes.
2. Capture checkerboard images and calibrate: 60 minutes.
3. Take 100 original object photographs: 45 minutes.
4. Undistort and manually label the images: 120 minutes.
5. Set up and train the segmentation model: 60 minutes of active work.
6. Test inference on a new photograph and check failure cases: 45 minutes.
7. Compare 10 test measurements with the calliper and test undistortion: 45 minutes.
8. Test edge positioning and measurement refusals: 30 minutes.
9. Update results, record the Loom video and open the pull request: 45 minutes.

I will use installation and training wait time for other tasks when possible. My priority is to get calibration, dataset, model and inference working before starting the extra experiments. If time runs out, I will record what I could not finish.

## Decisions

I fixed the image lists before collection: 70 for training, 20 for validation and 10 for testing. I will choose model settings using validation results, never the test set.

I chose a small U-Net for segmentation. I considered Mask R-CNN, but I only need one card mask, so the added complexity does not seem worthwhile. The simpler model will not separate multiple cards in one image.

I chose an ArUco marker instead of using one pixels per millimetre ratio for the entire image. It takes extra care to print, measure and detect, but its corners allow perspective correction when the card and marker lie on the same plane.


## Changes

### 4 November 2026: Dataset and calibration

I used a Samsung Galaxy S10+ and a checkerboard with 9 by 7 inner corners and 18 mm squares. I collected 35 usable calibration photographs and saved the camera parameters. The calibration reprojection error was about 2.363 pixels.

I used a joker playing card rather than the plastic card mentioned in the original plan. Its reference dimensions are 57 mm by 87 mm. The ArUco marker is dictionary 4x4_50, ID 0, with a 40 mm printed square. Both must lie flat on the same surface.

I collected and manually labelled 89 photographs rather than the planned 100. Before training, I removed the missing image names from the preassigned split lists rather than moving photographs between sets. This left 60 training, 19 validation and 10 test photographs. The model was selected using validation results only. No test photograph was used to tune the model.

### 5 November 2026: Model and measurement results

The small U-Net was trained on my labelled images. Its best checkpoint was selected at epoch 20 using a validation mask IoU of 0.9700. On the held-out segmentation test split, the mean mask IoU was 0.9688. I kept this checkpoint for the later measurement experiments.

I also took 10 new photographs to assess physical dimensions, separate from the original segmentation test split. On the first run, all 10 were accepted, with a mean absolute dimension error of 2.015 mm. This narrowly missed my original target of at most 2.0 mm. I kept the target unchanged.

I compared measurements from raw and undistorted versions of the same photographs. After the final refusal check, nine images were accepted in both conditions. Their mean error was 3.272 mm without correction and 1.964 mm with correction, a reduction of 1.308 mm, or about 40.0%. The segmentation model was trained on undistorted images, so this comparison reflects the complete pipeline rather than lens correction in isolation.

The speed benchmark used two warmup runs and five timed runs after model loading. The median processing time was 452.26 ms, below the original 1500 ms target. The recorded times ranged from 419.12 ms to 703.83 ms.

### 5 November 2026: Boundary failure and safety change

I tested four card positions moving toward the left image boundary. The first three were measured. The fourth card was partially cut off, but the original pipeline still returned 48.09 mm by 89.24 mm at 0.9463 mask confidence. This was not a trustworthy measurement, even though the segmentation confidence was high.

The original boundary check examined only the outermost mask pixels. The predicted mask stopped slightly short of the image edge, so the check missed the crop. I changed it to refuse a measurement if any mask pixels fall within 15 pixels of a processed image boundary. This is a conservative rule, not a guarantee that every cropped object will be detected. It may refuse some fully visible objects close to an edge.

After the change, the partially cropped stress photograph was refused and the fully visible photograph near the edge was still measured. I reran the 10-photo accuracy experiment: nine were accepted and one was refused. The mean absolute error on the nine accepted photographs was 1.964 mm, with a median of 2.228 mm and a worst case of 4.302 mm.

This meets the original accuracy objective of at most 2.0 mm with at least nine accepted photographs, but the lower mean reflects refusing one photograph, not an improvement in the accepted dimensions themselves. I retained the before-fix results for comparison.

### Remaining work

I still need to finalize the evidence and reproducibility documentation, record the walkthrough, and open the pull request from dev-test01 into main. I will keep any incomplete item visible in the Definition of Done checklist.

