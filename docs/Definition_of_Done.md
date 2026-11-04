# Definition of Done

I will update this checklist as the project progresses. I will only mark something complete when I have the actual file, number or link to support it. A script that runs is not proof that the measurements are accurate.

1. [ ] The first commit contains the Plan and fixed split lists, before any code. Commit link: pending.

2. [ ] I have at least 20 usable checkerboard photographs. The camera matrix, distortion coefficients and reprojection error are saved. Evidence: `calibration/parameters.npz`, calibration image count and output in `docs/Objectives.md`.

3. [ ] I have taken at least 70 photographs myself, kept the original files and labelled the undistorted images. My planned dataset has 100 images with 70 for training, 20 for validation and 10 for testing. Evidence: `dataset/raw/`, `dataset/images/`, `dataset/masks/` and `dataset/splits/`.

4. [ ] The test split was fixed before training and was not used to tune the model or choose a checkpoint. Evidence: initial commit and training configuration.

5. [ ] I have trained an allowed segmentation model on my images and reported test mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall. Evidence: training history, saved model and `models/checkpoints/test_metrics.json`.

6. [ ] One command can take a new raw photograph, undistort it, find the card and reference marker, and produce width, height, confidence and an overlay. Evidence: saved output and the Loom demonstration.

7. [ ] I have demonstrated what the system does when the card is missing and when the marker is missing. Neither case reports a trusted dimension. Evidence: saved inference results.

8. [ ] At least 10 test photographs have corresponding hand measurements. I have saved each error and reported the mean, median, worst case and percentage error. Evidence: `measurement/ground_truth.csv` and `measurement/results/`.

9. [ ] I have measured the same photographs with and without undistortion and compared the errors. Evidence: the raw and corrected results in `measurement/results/summary.json`.

10. [ ] MO-1 and MO-2 have measured results, including five timing runs. I have recorded whether each target was met. Evidence: `docs/Objectives.md` and the saved experiment outputs.

11. [ ] I have tested the card at different distances from the image centre and reported how error and refusal rate changed. Evidence: the edge experiment results.

12. [ ] I have written down when the system refuses to measure, why it refuses, and how many photographs are affected. Evidence: inference outputs and measurement summaries.

13. [ ] The Plan contains a decision I made, an alternative I rejected and the cost of that choice. Any later changes are recorded with the reason. Evidence: `docs/Plan.md`.

14. [ ] The README explains setup and commands, contains the hardware details and dataset summary, and links to the model weights with their SHA-256 hash. The code and dependency versions are recorded so someone else can run it.

15. [ ] The Loom video is no longer than five minutes. It shows a new photograph, the command, a physical measurement, the relevant code and answers to K1, K2, K3 and K4. Loom link: pending.

16. [ ] The work is in a public GitHub repository on `dev-test01`, with meaningful commits and a pull request into `main`. Pull request link: pending.

## What I did not finish

I will fill this section in at the end. Any incomplete requirement will stay unchecked, with a short explanation and any partial evidence I have.
