# AI_TECH_PROJECT

## Project Overview

This project is about **parasitic egg detection from microscopic images** using object detection models.  
The repository includes:

- training notebooks for **Google Colab**
- local Python scripts for **VS Code / terminal / desktop GUI**
- evaluation tools for comparing predictions with **COCO-format ground truth**
- model-building code for **Faster R-CNN** and **Cascade Faster R-CNN**

Even though the repository contains YOLO, Faster R-CNN, and Cascade Faster R-CNN, the **main model used in this project is Cascade Faster R-CNN**.  
So in this README, the workflow is written around **Cascade Faster R-CNN as the main pipeline**, and the other files are explained as supporting files for comparison, baseline testing, or extra experiments.

---

## Main Idea of the Whole Repository

The repository can be understood as **two connected parts**.

### Part 1: Training and experiments in Google Colab
These files are notebooks:

- `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`

These notebooks are mainly used to:

- mount Google Drive
- copy dataset zip files into Colab
- prepare train / validation / test data
- train the model
- save checkpoints
- run evaluation / plotting / extra analysis

### Part 2: Local testing and evaluation in VS Code
These files are Python scripts/modules:

- `app_v2.py`
- `eval_all_v2.py`
- `evaluator_v2.py`
- `models_v2.py`
- `requirements.txt`

These local files are mainly used to:

- load trained weights on another machine
- inspect predictions with a GUI
- run batch evaluation from the terminal
- compute confusion matrix, per-class metrics, mAP, mIoU, macro F1
- rebuild TorchVision-based detectors correctly before loading checkpoints

So the full project flow is:

1. **Train in Colab**
2. **Save checkpoint**
3. **Download checkpoint to local machine**
4. **Use `app_v2.py` to inspect predictions visually**
5. **Use `eval_all_v2.py` to run full evaluation**
6. `eval_all_v2.py` internally uses `models_v2.py` and `evaluator_v2.py`

That is the main connection between all files.

---

## Recommended Platform for Each File

## Use in Google Colab

These files are better suited for Colab:

- `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`

Why:
- they are notebook-based
- they usually install packages inside notebook cells
- they are organized like experiment notebooks
- they often use Google Drive paths and Colab local paths
- they are more convenient for GPU training

## Use in VS Code / local Python

These files are better suited for local development:

- `app_v2.py`
- `eval_all_v2.py`
- `evaluator_v2.py`
- `models_v2.py`
- `requirements.txt`

Why:
- they are normal Python files
- they can be run from terminal or VS Code
- they are used for inspection, evaluation, and model loading on a local machine

---

## Repository Structure

```bash
.
├── AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
├── AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb
├── AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
├── requirements.txt
└── README.md
```

---

## Recommended Workflow for This Project

Because **Cascade Faster R-CNN is the main model**, the cleanest workflow is:

### Step 1 — Train the main model in Colab
Use:

```bash
AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
```

This notebook is the main training notebook for the project.

### Step 2 — Save the trained checkpoint
After training, save the model checkpoint file, for example:

- `cascade_frcnn_best.pth`
- `best_cascade.pth`
- or any checkpoint name you used

### Step 3 — Download the checkpoint to your local machine
Put the checkpoint inside your local project folder, for example:

```bash
project/
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
├── requirements.txt
├── weights/
│   └── cascade_frcnn_best.pth
├── data/
│   ├── images/
│   └── test.json
└── classes.txt
```

### Step 4 — Open the model in the GUI
Run:

```bash
python app_v2.py
```

Use the GUI to:
- load the Cascade checkpoint
- choose the image folder
- choose the COCO ground-truth JSON
- run prediction on all test images
- inspect results one by one
- export CSV
- save failed cases
- display a confusion matrix

### Step 5 — Run full batch evaluation
Run:

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --cascade_best weights/cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_cascade_main
```

This is the main evaluation command if Cascade Faster R-CNN is your primary model.

---

## Setup on Another Machine

If someone downloads this project and wants to run it on another computer, these are the recommended steps.

## 1) Install Python

Recommended version:

- **Python 3.10** or
- **Python 3.11**

Using these versions is safer because PyTorch, torchvision, PyQt5, and ultralytics are commonly used with them.

---

## 2) Download / copy all required files

At minimum, copy these files:

- `app_v2.py`
- `eval_all_v2.py`
- `evaluator_v2.py`
- `models_v2.py`
- `requirements.txt`

If training is also needed, copy the notebooks too:

- `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`

---

## 3) Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4) Install required packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Important note about PyTorch and CUDA

If the local machine has an NVIDIA GPU, it is often better to install `torch` and `torchvision` using the official PyTorch command that matches the installed CUDA version.

Example idea:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

But if CPU-only is fine, the normal `requirements.txt` installation is usually enough.

---

## 6) Prepare the dataset

For local evaluation, you need:

- an image folder
- a COCO-format JSON annotation file
- the model checkpoint
- optionally `classes.txt`

Example:

```bash
project/
├── data/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── test.json
├── weights/
│   └── cascade_frcnn_best.pth
├── classes.txt
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
└── requirements.txt
```

---

## 7) About `classes.txt`

For TorchVision-based models such as:

- Faster R-CNN
- Cascade Faster R-CNN

the script needs to know the number of classes so it can rebuild the detector correctly before loading weights.

You can provide that in one of two ways:

### Option A — use `classes.txt`
One class name per line:

```txt
class_1
class_2
class_3
...
```

### Option B — use `--num_classes`
If you know the number of object classes directly, you can pass it from the command line.

Important note:
- in the code, TorchVision models are created with **background included**
- so if your dataset has `N` object classes, the internal model may use `N + 1`

The script already handles this logic.

---

## 8) Path setup

There are two different path situations in this repository.

### A. In Colab notebooks
Paths are usually things like:
- `/content/...`
- `/content/drive/MyDrive/...`

These paths are for Google Colab only.

### B. In local Python scripts
Paths should be local machine paths such as:
- `data/images`
- `data/test.json`
- `weights/cascade_frcnn_best.pth`
- `classes.txt`

So when moving from Colab to local machine, you must change the paths.

---

## Main Files and What Each One Does

This section explains all 8 files in the project and how they connect.

---

# 1) `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`

## Main role
This is the **main training notebook** of the project.  
If your project focuses on **Cascade Faster R-CNN**, this is the most important notebook.

It is used to:
- prepare the dataset
- create train / validation / test split
- define the custom Cascade Faster R-CNN model
- train the model
- save checkpoints
- run evaluation and extra analysis after training

## Recommended platform
- **Google Colab**

## What this notebook is mainly for
Use this notebook when you want to:
- train the main detector
- continue training
- evaluate a saved Cascade checkpoint in notebook form
- inspect confusion matrix, PR-style statistics, and mIoU-related results

## Main functions / classes inside

### `prepare_local_data()`
Copies or prepares dataset files into a local working directory for training.

### `find_coco_json()`
Searches for the correct COCO annotation file.

### `collate_batch()`
Custom collate function for the DataLoader so batches of detection data can be loaded correctly.

### `EggCocoDataset`
Custom dataset class for reading images and COCO annotations in a format suitable for TorchVision detection models.

### `encode_boxes()`
Encodes ground-truth boxes relative to proposals for box regression.

### `decode_boxes()`
Decodes predicted box deltas back into real bounding boxes.

### `TwoFCHead`
Defines the two fully connected layers used in the RoI head.

### `ClassAgnosticPredictor`
Defines a class-agnostic box regressor with class prediction output.

### `subsample_labels()`
Samples positive and negative proposals for training.

### `fastrcnn_loss_agnostic()`
Computes classification loss and class-agnostic box regression loss.

### `CascadeRoIHeads`
This is one of the most important custom parts in the notebook.  
It implements the **multi-stage Cascade RoI head** used by the model.

### `fix_coco_gt()`
Repairs missing COCO fields such as `area` or `iscrowd` when needed.

### `coco_evaluate()`
Runs COCO evaluation logic on saved detection results.

### `train_one_epoch()`
Trains the model for one epoch.

### `save_ckpt()`
Saves model checkpoints.

### `try_resume()`
Loads previous checkpoint information and resumes training if available.

### `save_model_to_drive()`
Saves model/checkpoint outputs back to Google Drive.

### `cuda_sync()`
Synchronizes CUDA timing when measuring inference speed.

### `timed_frcnn_predict()`
Runs a prediction pass while measuring time.

### `compute_per_class_ious()`
Computes IoU statistics per class.

### `frcnn_label_to_classidx()`
Maps model labels into class indices.

### `load_frcnn_ckpt()`
Loads a saved FRCNN/Cascade checkpoint.

### `eval_test_frcnn_miou_and_time()`
Evaluates mean IoU and inference time on the test set.

### `plot_cm()`
Plots a confusion matrix.

### `pr_and_cm_from_detections()`
Computes precision/recall style matching and confusion matrix values from detections.

### `coco_eval_from_results()`
Runs COCO mAP evaluation from a saved result structure.

### `eval_frcnn_on_test()`
Full test evaluation helper for FRCNN-style models.

## Summary of this notebook
This notebook is the **center of the Cascade Faster R-CNN pipeline**.  
If you only keep one training notebook as the main one, this should be the one.

---

# 2) `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`

## Main role
This notebook is the **Faster R-CNN baseline notebook**.

It is useful when you want to:
- compare Cascade Faster R-CNN against normal Faster R-CNN
- test a simpler two-stage detector
- measure whether the cascade design really improves the result

## Recommended platform
- **Google Colab**

## Main functions / classes inside

### `prepare_local_data()`
Prepares local dataset files.

### `find_coco_json()`
Finds the correct COCO JSON annotation file.

### `collate_batch()`
Custom batch collation for detection data.

### `EggCocoDataset`
Reads image and annotation data for training/evaluation.

### `train_one_epoch()`
Runs one training epoch for Faster R-CNN.

### `coco_evaluate()`
Runs COCO evaluation.

### `fix_coco_gt()`
Repairs missing COCO fields if needed.

### `detection_confusion_matrix()`
Builds a confusion-matrix-style view for detections.

### `map_yolo_cls_to_classidx()`
Maps YOLO class IDs into the shared class-index format for comparisons.

### `frcnn_label_to_classidx()`
Maps Faster R-CNN labels into class indices.

### `load_frcnn_ckpt()`
Loads a saved Faster R-CNN checkpoint.

### `compute_image_ious_per_class()`
Computes IoU per image and per class.

### `eval_miou_yolo()`
Evaluates mean IoU for YOLO predictions.

### `eval_miou_frcnn()`
Evaluates mean IoU for Faster R-CNN predictions.

### `cuda_sync()`
Synchronizes CUDA for timing.

### `timed_yolo_predict()`
Measures YOLO inference time.

### `timed_frcnn_predict()`
Measures Faster R-CNN inference time.

### `compute_per_class_ious()`
Computes per-class IoU.

### `eval_test_yolo_miou_and_time()`
Evaluates YOLO test performance with mIoU and timing.

### `eval_test_frcnn_miou_and_time()`
Evaluates Faster R-CNN test performance with mIoU and timing.

### `plot_cm()`
Plots confusion matrix.

### `pr_and_cm_from_detections()`
Computes confusion matrix and PR-style matching data from detections.

### `coco_eval_from_results()`
Runs COCO evaluation from result data.

### `yolo_results_csv_path()`
Helps locate YOLO result CSV files.

### `read_yolo_best_losses()`
Reads YOLO training history values from CSV/log files.

### `eval_yolo_on_test()`
Evaluates YOLO on the test set.

### `eval_frcnn_on_test()`
Evaluates Faster R-CNN on the test set.

### `save_cm_csv()`
Saves confusion matrix values to CSV.

### `plot_cm_numbers()`
Plots confusion matrix with numeric values.

### `top_confusions()`
Finds the most frequent confusion pairs.

## Summary of this notebook
This notebook is useful as a **baseline comparison notebook**.  
It is not the main project notebook, but it is important if you want to show that Cascade Faster R-CNN performs better than standard Faster R-CNN.

---

# 3) `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`

## Main role
This notebook is the **YOLO training notebook**.

It is mainly used for:
- fast training experiments
- quick baseline comparison
- converting COCO labels into YOLO label format
- testing how a one-stage detector behaves on the dataset

## Recommended platform
- **Google Colab**

## Main functions inside

### `prepare_local_copy()`
Prepares local copies of the dataset in Colab.

### `coco_to_yolo_txt()`
Converts COCO annotations into YOLO `.txt` label format.

### `prepare_datasets()`
Builds the dataset structure required for YOLO training.

### `run_selected_training()`
Starts YOLO training using the selected model/configuration.

## Summary of this notebook
This notebook is useful for **comparison and baseline speed experiments**.  
It is not the main notebook if your project focuses on Cascade Faster R-CNN, but it is still helpful if you want to compare one-stage and two-stage detection.

---

# 4) `app_v2.py`

## Main role
This file is the **main local desktop GUI** for checking model predictions visually.

Because this project uses **Cascade Faster R-CNN as the main model**, `app_v2.py` should be treated as the local review application for:

- loading a trained Cascade/Faster R-CNN checkpoint
- selecting the image folder
- selecting the COCO ground-truth JSON
- running prediction on the local test set
- checking result images one by one
- exporting CSV summary
- saving failure cases
- displaying confusion matrix

## Recommended platform
- **VS Code**
- or any local Python environment with desktop GUI support

## How it connects to the project
This file is normally used **after training is already finished**.

Typical use:
1. train the model in `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`
2. save the best checkpoint
3. download the checkpoint
4. run `app_v2.py`
5. visually inspect how the checkpoint behaves on the test set

## Main helper functions / classes inside

### `to_qimage()`
Converts an OpenCV BGR image into a `QImage` so it can be shown in the PyQt interface.

### `draw_detection_boxes()`
Draws predicted or ground-truth boxes and labels on an image.

### `compare_prediction_with_gt()`
Compares predicted boxes against ground truth using IoU and class matching, then returns TP/FP/FN style results.

### `ensure_coco_fields()`
Adds missing COCO fields such as `iscrowd` or `area` if they are missing.

### `normalize_int_mapping()`
Normalizes mapping dictionaries so keys/values are integers.

### `encode_box_targets()`
Encodes regression targets for proposal refinement.

### `decode_box_deltas()`
Decodes box deltas back into coordinate boxes.

### `TwoFCHead`
RoI fully connected head.

### `ClassAgnosticPredictor`
Class-agnostic box regression predictor with class logits.

### `sample_training_labels()`
Samples positive and negative proposals.

### `agnostic_fastrcnn_loss()`
Loss function for class-agnostic FRCNN-style box regression.

### `CascadeRoIHeads`
Custom Cascade RoI heads used to rebuild the Cascade detector.

### `build_cascade_detector()`
Builds the custom Cascade model architecture.

### `load_checkpoint_and_build_model()`
Loads the checkpoint and rebuilds the correct model structure based on the dataset/classes.

### `PreviewDialog`
A small preview dialog used inside the GUI.

### `CascadeReviewApp`
The main PyQt application window.

## Important GUI methods inside `CascadeReviewApp`

### `run_inference_on_image()`
Runs prediction on a single image.

### `run_all_predictions()`
Runs prediction for the full selected image folder and stores the results.

### `show_prev()` / `show_next()`
Moves between evaluated images.

### `export_csv()`
Exports result summary to CSV.

### `save_fail_cases()`
Saves incorrect images to a folder for manual review.

### `show_confusion_matrix()`
Shows a confusion matrix built from the current evaluation result.

### `main()`
Starts the PyQt application.

## When to use this file
Use `app_v2.py` when you want:
- interactive checking
- image-by-image inspection
- debugging wrong detections
- screenshots for reports/presentations
- manual review of failure cases

---

# 5) `eval_all_v2.py`

## Main role
This file is the **main command-line evaluation script**.

It evaluates trained models on a COCO-format dataset and writes the outputs into:

```bash
runs_eval/<run_name>/
```

Even though it supports multiple model types, for this project the main usage is usually:

- `--cascade_best` for the main model
- optionally `--frcnn_best` or YOLO weights for comparison

## How it connects to the project
This file is normally used after you already have:

- test images
- COCO JSON
- model weights
- class information

It calls:
- `models_v2.py` to rebuild and load the detector
- `evaluator_v2.py` to calculate all metrics

So the connection is:

```bash
eval_all_v2.py
    -> models_v2.py
    -> evaluator_v2.py
```

## Main functions inside

### `count_lines_in_class_file()`
Counts how many classes exist in `classes.txt`.

### `parse_args()`
Reads command-line arguments such as:
- `--coco_json`
- `--img_dir`
- `--cascade_best`
- `--frcnn_best`
- `--yolo8_best`
- `--yolo11_best`
- `--classes_txt`
- `--num_classes`
- `--score_thr`
- `--iou_thr`
- `--device`
- `--run_name`
- `--limit_images`

### `get_torchvision_num_classes()`
Converts the given class information into the correct number of classes for TorchVision models.

### `main()`
Runs the whole evaluation process.

## Recommended command for the main model
Because Cascade Faster R-CNN is the main detector, this is the most important command:

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --cascade_best weights/cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_cascade_main
```

## Optional comparison commands

### Faster R-CNN baseline

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --frcnn_best weights/faster_rcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_frcnn_baseline
```

### YOLO comparison

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --yolo8_best weights/best.pt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_yolov8_compare
```

## When to use this file
Use `eval_all_v2.py` when you want:
- final benchmark evaluation
- repeatable evaluation from terminal
- saved output files
- summary numbers for tables in reports

---

# 6) `evaluator_v2.py`

## Main role
This file is the **metric and evaluation backend** used by `eval_all_v2.py`.

It is responsible for:
- reading COCO annotations
- matching predictions and ground truth
- building confusion matrix
- computing precision, recall, F1
- computing mean IoU
- creating COCO JSON results
- running COCO mAP evaluation
- saving CSV/JSON outputs

Usually, you do not run this file directly.  
Instead, it is called by `eval_all_v2.py`.

## Main classes / functions inside

### `Prediction`
Small structure for storing:
- boxes
- scores
- labels

### `GroundTruth`
Small structure for storing:
- boxes
- labels

### `load_coco_info()`
Loads COCO annotations and builds mappings between category IDs and internal class labels.

### `get_image_path()`
Gets the correct image path from COCO image info and the selected image folder.

### `load_gt_for_image()`
Loads ground-truth boxes and labels for one image.

### `greedy_match()`
Matches predictions with ground truth using greedy IoU-based matching.

### `build_confusion_matrix()`
Builds a raw confusion matrix including background handling.

### `save_confusion_plot()`
Saves the confusion matrix plot as an image.

### `compute_class_metrics()`
Computes TP, FP, FN, precision, recall, F1, and mean IoU per class.

### `coco_map_from_json()`
Runs COCO mAP evaluation from a saved detection JSON file.

### `save_predictions_as_coco_json()`
Converts prediction results into a COCO-style detection JSON file.

### `run_full_evaluation()`
This is the main function of the file.  
It performs the full evaluation pipeline:
- inference over all images
- save detections JSON
- compute COCO mAP
- compute confusion matrix
- compute per-class metrics
- save plots and summaries
- return a final summary dictionary

## Output files produced through this module
Typical output files inside `runs_eval/<run_name>/<model_name>/`:

- `detections_coco.json`
- `confusion_matrix_counts.csv`
- `confusion_matrix.png`
- `per_class_metrics.csv`
- `metrics_summary.json`

---

# 7) `models_v2.py`

## Main role
This file contains **model-building and model-loading helpers**.

It is responsible for:
- wrapping YOLO prediction
- building Faster R-CNN
- building Cascade Faster R-CNN
- loading checkpoint weights
- running inference for local evaluation

This file is used by `eval_all_v2.py`, and indirectly supports `app_v2.py` because both need to rebuild or load the model correctly.

## Main classes / functions inside

### `YOLOPredictor`
Wrapper around Ultralytics YOLO.
It provides `predict_one()` in a unified format.

### `build_faster_rcnn()`
Builds a TorchVision Faster R-CNN model and replaces the default box predictor.

### `ClassAgnosticFastRCNNPredictor`
Defines a class-agnostic predictor.

### `decode_boxes()`
Decodes box deltas into actual bounding boxes.

### `refine_boxes()`
Refines proposal boxes and clips them to the image size.

### `TwoFCHead`
Defines the fully connected head for RoI features.

### `ClassAgnosticPredictor`
Defines the class and box predictor used in the cascade head.

### `CascadeRoIHeads`
Defines the custom Cascade Faster R-CNN RoI head.

### `build_cascade_frcnn()`
Builds the Cascade Faster R-CNN architecture.

### `DetectorWrapper`
Loads a checkpoint, moves the model to the correct device, and provides `predict_one()` for a single image.

## Important note
If the checkpoint was saved with:
- `model_state_dict`
or
- the raw state dict

the wrapper is written to handle both cases.

---

# 8) `requirements.txt`

## Main role
This file lists the Python packages needed to run the local scripts and support the notebooks.

Typical packages include:
- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `PyQt5`
- `torch`
- `torchvision`
- `ultralytics`
- `pycocotools`
- `tqdm`
- plus some helper libraries such as `Pillow`, `PyYAML`, `scikit-learn`, and `seaborn`

## How to use it

```bash
pip install -r requirements.txt
```

---

## How All Files Connect Together

This is the most important relationship in the repository.

### Training side
```bash
AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
    -> trains the main Cascade Faster R-CNN model
    -> saves the checkpoint
```

### Local inspection side
```bash
app_v2.py
    -> loads the trained checkpoint
    -> loads image folder + COCO JSON
    -> runs local visual inspection
```

### Local benchmark side
```bash
eval_all_v2.py
    -> uses models_v2.py to rebuild/load the detector
    -> uses evaluator_v2.py to compute metrics
    -> saves result files
```

### Comparison side
```bash
AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb
AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb
    -> train baseline / comparison models
    -> optional use with eval_all_v2.py
```

So if the project is centered on **Cascade Faster R-CNN**, the real core chain is:

```bash
AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
    -> trained checkpoint
    -> app_v2.py
    -> eval_all_v2.py
        -> models_v2.py
        -> evaluator_v2.py
```

---

## Example Local Folder Layout

A clean local project layout could look like this:

```bash
AI_TECH_PROJECT/
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
├── requirements.txt
├── classes.txt
├── data/
│   ├── images/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── test.json
├── weights/
│   ├── cascade_frcnn_best.pth
│   ├── faster_rcnn_best.pth
│   └── best.pt
├── AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
├── AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb
└── AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb
```

---

## How to Run the Main GUI

If your main model is Cascade Faster R-CNN:

```bash
python app_v2.py
```

Then in the GUI:

1. click **Choose model checkpoint**
2. select your Cascade checkpoint
3. click **Choose Image Folder**
4. choose the folder containing test images
5. click **Choose GT COCO json**
6. select `test.json`
7. click the run button
8. inspect the results

Main features:
- previous / next image browsing
- summary text
- predicted boxes vs ground truth
- export CSV
- save failed cases
- confusion matrix

---

## How to Run Final Evaluation for the Main Model

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --cascade_best weights/cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_cascade_main
```

### If you want CPU instead of GPU

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --cascade_best weights/cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cpu \
  --run_name eval_cascade_cpu
```

---

## Common Problems and Notes

## 1) `ModuleNotFoundError`
Make sure:
- the scripts are in the same folder
- your virtual environment is activated
- all packages from `requirements.txt` are installed

## 2) Checkpoint loads but some keys are missing
This can happen if:
- the class count does not match
- the architecture differs from the checkpoint used during training
- the checkpoint was saved from a slightly different version of the model

For Cascade/FRCNN, always make sure:
- `classes.txt` is correct
- the rebuilt model matches the training configuration

## 3) No images found
Make sure:
- `--img_dir` points to the real image folder
- file names in the COCO JSON match the image file names in the folder

## 4) GUI does not open
Possible reasons:
- PyQt5 is not installed
- the machine does not support GUI display properly
- remote/headless environment is being used

In that case, use `eval_all_v2.py` instead of the GUI.

## 5) `pycocotools` install issue on Windows
On some Windows environments, `pycocotools` can be harder to install.  
If that happens, use an alternative compatible build or install tools required for compilation.

## 6) Wrong class count
If evaluation fails or predictions look wrong, check:
- `classes.txt`
- category IDs in COCO JSON
- label mapping logic
- whether background is handled as `+1` internally for TorchVision models

---

## Final Recommendation

If this project is presented as a full pipeline, the cleanest explanation is:

- **Cascade Faster R-CNN is the main model**
- `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb` is the main training notebook
- `app_v2.py` is the local GUI for visual checking
- `eval_all_v2.py` is the final benchmark script
- `models_v2.py` rebuilds/loads the model
- `evaluator_v2.py` computes metrics
- `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb` and `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb` are supporting notebooks for comparison/baseline experiments

That gives a complete engineering workflow from training to local evaluation.

---

## Short Version of the Main Usage

### Train the main model
Use:

```bash
AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
```

### Inspect prediction visually
Use:

```bash
python app_v2.py
```

### Run final evaluation
Use:

```bash
python eval_all_v2.py \
  --coco_json data/test.json \
  --img_dir data/images \
  --cascade_best weights/cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_cascade_main
```
