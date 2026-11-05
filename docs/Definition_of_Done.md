
# Definition of Done

This checklist records what I have completed and where the evidence can be found. I have left the remaining submission tasks unchecked.

## Completed requirements

1. [x] The project began with a written plan, objectives and fixed dataset split lists. These were committed before collecting data and training the model. Evidence: initial repository commits and `docs/`.

2. [x] I calibrated the camera using 35 checkerboard photographs. The checkerboard had 9 by 7 inner corners and 18 mm squares. I saved the camera matrix, distortion coefficients and reprojection errors. The reprojection error was approximately 2.363 pixels. Evidence: `calibration/images/`, `calibration/parameters.npz` and `calibration/calibration_views.csv`.

3. [x] I collected 89 original photographs of a playing card and manually created 89 segmentation masks. The original images and processed images have been retained. Evidence: `dataset/raw/`, `dataset/images/`, `dataset/masks/` and `dataset/image_manifest.csv`.

4. [x] I fixed the split assignments before training and did not use the test split for model selection. After removing missing image names from the original plan, the dataset contained 60 training, 19 validation and 10 test images. Evidence: `dataset/splits/` and `models/training_config.json`.

5. [x] I trained a small U-Net segmentation model using my labelled images. The best checkpoint was selected at epoch 20 using validation IoU. On the untouched segmentation test split, mean mask IoU was 0.9688, mAP@0.5 was 1.0000, mAP@0.5:0.95 was 1.0000, pixel precision was 0.9777 and pixel recall was 0.9919. The AP calculation uses one proposed object per image, as documented in the evaluation output. Evidence: `models/train.py`, `models/best_unet.pth` and `models/evaluation/test_metrics.json`.

6. [x] A single command processes a new photograph, applies calibration, segments the card, detects the ArUco reference and calculates dimensions in millimetres. It saves a labelled overlay and a JSON result. Evidence: `inference/measure.py` and `inference/outputs/`.

7. [x] I tested missing-object and missing-reference cases. The system refused both without returning dimensions. Evidence: `inference/new_images/`, `inference/outputs/no_object_01_overlay.json` and `inference/outputs/no_reference_01_overlay.json`.

8. [x] I evaluated physical measurement accuracy on 10 new photographs of the same card, whose reference dimensions were 57 by 87 mm. After the final boundary check, nine measurements were accepted and one was refused. The mean absolute dimension error on accepted photographs was 1.964 mm, the median was 2.228 mm, the worst case was 4.302 mm and the mean percentage error was 2.852%. Evidence: `measurement/images/`, `measurement/results/per_instance.csv` and `measurement/results/summary.json`.

9. [x] I compared raw and undistorted versions of the same 10 photographs. Nine were accepted under both conditions. The mean error was 3.272 mm without undistortion and 1.964 mm with undistortion, a reduction of 1.308 mm, or approximately 40.0%. This is a whole-pipeline comparison because the segmentation model was trained on undistorted images. Evidence: `measurement/results/undistortion_comparison.csv` and `measurement/results/undistortion_summary.json`.

10. [x] Both measurable objectives have been evaluated. MO-1 required mean error no greater than 2.0 mm with at least nine of 10 photographs accepted; the final results met both conditions. MO-2 required median processing time no greater than 1500 ms; the measured median was 452.26 ms over five runs after two warmups. Evidence: `docs/Objectives.md`, `measurement/results/summary.json` and `measurement/results/benchmark.json`.

11. [x] I tested the card at four positions progressively closer to the image boundary. The first three were measured with mean dimension errors of approximately 1.96 mm, 1.63 mm and 2.41 mm. The fourth was partially outside the frame and was incorrectly measured by the original pipeline. After changing the boundary check, that photograph was correctly refused. Evidence: `measurement/stress_images/`, `inference/outputs/stress_01_overlay.json` through `stress_04_overlay.json`, and the change record in `docs/Plan.md`.

12. [x] I defined refusal conditions for missing objects, missing references, low mask confidence, insufficient mask area, unusable markers and objects near the image boundary. The final accuracy experiment refused one of 10 photographs. The boundary stress test refused one of four photographs after the correction. Refusals are reported separately from accepted measurement errors. Evidence: `inference/measure.py` and the saved experiment results.

13. [x] I documented engineering decisions, including choosing U-Net over a more complex instance segmentation approach, using an ArUco marker instead of a global pixel scale, retaining the original accuracy target and adding a conservative boundary margin after observing a failure. Evidence: `docs/Plan.md`.

## Remaining submission requirements

14. [x] Finalize the README with setup instructions, dataset and hardware details, commands, reproducibility information, model weight availability and SHA-256 checksum.

15. [x] Record a Loom walkthrough no longer than five minutes. Demonstrate a new photograph, the measurement command, physical dimensions, relevant code and the four required discussion points (K1 to K4). Add the Loom link to the README.

16. [x] Verify the final repository contents, commit and push the remaining work to `dev-test01`, and open a public pull request into `main`. Add the pull request link here.

## Limitations and incomplete work

The model was trained to segment one specific playing card, not arbitrary household objects. The dimensional accuracy experiment used repeated photographs of that same card rather than 10 different physical objects.

The accuracy objective was met after adding the boundary refusal check. The mean error decreased partly because one photograph was refused, not because the model's dimensional predictions improved.

The 15-pixel boundary margin is a heuristic. It can reject valid objects near the edge and cannot guarantee that every cropped or occluded object will be detected.

The remaining documentation and submission items will stay unchecked until their evidence is available.
