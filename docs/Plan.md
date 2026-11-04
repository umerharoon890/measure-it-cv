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

None yet. I will add dated notes here if the approach changes, rather than rewriting the original plan.
