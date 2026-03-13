
# AI_TECH_PROJECT

## Project overview

This project is about **parasitic egg detection from microscopic images** using multiple object detection approaches:

- **YOLO** for training and fast inference
- **Faster R-CNN** as a two-stage baseline
- **Cascade Faster R-CNN** as a custom two-stage detector with multi-stage box refinement

The repository is split into **two working parts**:

1. **Training / experiment notebooks (`.ipynb`)**  
   These are mainly for **Google Colab** because they mount Google Drive, install packages inside cells, unzip datasets from Drive, train models, save checkpoints, and run evaluation plots.

2. **Local Python files (`.py`)**  
   These are mainly for **VS Code or any local Python environment** on Windows/Linux/macOS.  
   They are used for:
   - desktop GUI review
   - command-line evaluation
   - model rebuilding/loading
   - metric computation and export

---

## Recommended platform for each file

### Use in **Google Colab**
These files are training notebooks and depend on Colab-style workflow:

- `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`
- `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`
- `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`

Why Colab is recommended:
- the notebooks use `drive.mount('/content/drive')`
- they install packages inside cells with `!pip install ...`
- they copy zip datasets from Google Drive to local Colab storage
- they are written like experiment notebooks, not production scripts

### Use in **VS Code / local Python**
These files are local scripts/modules:

- `app_v2.py`
- `eval_all_v2.py`
- `evaluator_v2.py`
- `models_v2.py`
- `requirements.txt`

Recommended local setup:
- **VS Code**
- **Python 3.10 or 3.11**
- optional **CUDA GPU** if you want faster inference/evaluation for PyTorch models

---

## Repository structure

```bash
.
├── AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb
├── AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb
├── AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
├── requirements.txt
└── README.md
```

---

## Full workflow of the project

A simple way to understand the repository is:

### Part A: Train models in Colab
Use the notebooks to:
- mount Drive
- copy dataset zip files from Drive
- unpack them to Colab local storage
- prepare train/validation/test data
- train the selected detector
- save checkpoints and logs
- run extra evaluation/analysis cells

### Part B: Use trained weights on another machine
After training is done, download or copy the trained weight files (`.pt` / `.pth`) to your local machine.

### Part C: Evaluate or inspect locally
Then use local Python files to:
- open the desktop review GUI (`app_v2.py`)
- run batch evaluation from terminal (`eval_all_v2.py`)
- compute metrics (`evaluator_v2.py`)
- rebuild/load detector architectures (`models_v2.py`)

---

## What you need before running the project on another machine

Before running anything locally, prepare these items first:

### 1. Python
Install:
- **Python 3.10** or **Python 3.11**

### 2. Code files
Copy these files into the same project folder:
- `app_v2.py`
- `eval_all_v2.py`
- `evaluator_v2.py`
- `models_v2.py`
- `requirements.txt`

### 3. Dataset
Prepare:
- a folder containing the test images
- a **COCO-format JSON** annotation file for the same image set

### 4. Model weights
Prepare whichever model you want to use:
- YOLO `.pt`
- Faster R-CNN `.pth`
- Cascade Faster R-CNN `.pth` or `.pt` depending on how you saved it

### 5. Optional class list file
For Faster R-CNN and Cascade Faster R-CNN evaluation, you will usually need either:
- `classes.txt`, or
- `--num_classes`

because the script must rebuild the TorchVision detector with the correct number of classes.

---

## Important setup note before running locally

### File import names must match

In the current local scripts, the imports may still point to earlier filenames such as:

- `evaluator_student_style_v2`
- `models_student_style_v2`

If your actual files are named:
- `evaluator_v2.py`
- `models_v2.py`

then you must do **one** of these:

#### Option 1: rename the files
Rename:
- `evaluator_v2.py` → `evaluator_student_style_v2.py`
- `models_v2.py` → `models_student_style_v2.py`

#### Option 2: edit the import lines
Update the imports inside the files so they match your current filenames.

For example:

```python
# in eval_all_v2.py
from evaluator_v2 import run_full_evaluation
from models_v2 import DetectorWrapper, YOLOPredictor, build_cascade_frcnn, build_faster_rcnn

# in models_v2.py
from evaluator_v2 import Prediction
```

This is very important. If you do not fix the import names first, the local scripts will fail with `ModuleNotFoundError`.

---

## Installation on another machine

### 1. Open the project folder
Open the folder in **VS Code**.

### 2. Create a virtual environment

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. PyTorch note
If you want GPU support, sometimes it is better to install `torch` and `torchvision` from the official PyTorch command for your CUDA version instead of only relying on `requirements.txt`.

---

## requirements.txt

The current requirements file includes the main packages used across the repository:

```txt
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
opencv-python>=4.8
Pillow>=10.0
scikit-learn>=1.3
tqdm>=4.66
PyYAML>=6.0
seaborn>=0.13
ultralytics>=8.2
PyQt5>=5.15
pycocotools>=2.0.7
torch>=2.1
torchvision>=0.16
```

---

## Expected dataset format

### For local evaluation (`app_v2.py` and `eval_all_v2.py`)
You should have something like this:

```bash
project_folder/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── test.json
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
└── requirements.txt
```

Where:
- `images/` contains the actual image files
- `test.json` is a valid **COCO detection annotation file**
- image file names inside JSON must match the real files in `images/`

### For notebook training
The notebooks are written for zip-based data copied from Google Drive.  
Typical structure inside the zip should include:
- images
- COCO JSON annotations
- category definitions

The notebooks search for annotation JSON files automatically, then split or convert them as needed.

---

## Detailed explanation of each file

# 1) `AI_TECH_PROJECT_TRAINING_YOLO_v2.ipynb`

## Purpose
This notebook is used to train the **YOLO** model in **Google Colab**.

## Main job of this notebook
It handles the full YOLO training pipeline:
- mount Google Drive
- install YOLO-related packages
- copy dataset zip files from Drive to local Colab storage
- unpack the dataset
- convert COCO annotations to YOLO `.txt` labels
- create train/validation/test splits
- generate the dataset YAML file
- train the selected YOLO model
- plot training results

## Main sections inside the notebook

### Setup
- mounts Google Drive
- installs `ultralytics`, `seaborn`, `matplotlib`, `pandas`

### Paths and dataset prep
- defines dataset zip path on Drive
- defines local temp folder
- defines experiment output folder

### Label conversion and split
- converts COCO boxes into YOLO text files
- writes split lists
- writes YAML config used by Ultralytics

### Training and summary
- selects YOLO model
- runs training
- logs results
- plots final graphs

## Important functions

### `prepare_local_copy()`
Copies the zip files from Google Drive into local Colab storage and extracts them.

### `coco_to_yolo_txt(json_path, img_dir, label_dir, prefix="")`
Reads a COCO JSON file and converts each annotation into YOLO text format.

### `prepare_datasets()`
Builds the train/validation/test structure for YOLO training and writes dataset config files.

### `run_selected_training()`
Runs training for the selected YOLO model(s), saves outputs, and records results.

## When to use this file
Use this notebook when you want to:
- train YOLO from scratch or fine-tune it
- rebuild YOLO labels from COCO format
- compare YOLO experiments in Colab

---

# 2) `AI_TECH_PROJECT_TRAINING_FASTER_RCNN_v2.ipynb`

## Purpose
This notebook is used to train and analyze the **Faster R-CNN** baseline in **Google Colab**.

## Main job of this notebook
It handles:
- dataset copy/unzip from Drive
- train/validation split
- dataset/dataloader creation
- Faster R-CNN model building
- training loop
- validation using COCO metrics
- checkpoint/history saving
- extra analysis such as confusion matrix, mIoU, and timing

## Main sections inside the notebook

### Setup
- mounts Drive
- installs PyTorch-related utilities and COCO tools

### Data split and loading
- finds annotation files
- creates train/validation JSON files
- builds a PyTorch dataset and dataloaders

### Model and training
- builds a Faster R-CNN model
- replaces the classification head
- trains the model
- saves logs/checkpoints

### Extra analysis
- confusion matrix
- mIoU comparison
- timing/inference analysis
- test evaluation helpers

## Important functions

### `prepare_local_data()`
Copies training data from Drive to local storage and unpacks it.

### `find_coco_json(root_dir)`
Searches for a valid COCO JSON annotation file.

### `collate_batch(batch)`
Custom collate function for variable-length detection targets.

### `EggCocoDataset`
PyTorch dataset class that loads images and COCO annotations.

### `train_one_epoch(...)`
Runs one training epoch for Faster R-CNN.

### `coco_evaluate(...)`
Runs validation with COCO-style evaluation.

### `fix_coco_gt(...)`
Cleans or rewrites ground-truth JSON so COCO evaluation works correctly.

### `detection_confusion_matrix(...)`
Builds a confusion matrix for object detection results.

### `load_frcnn_ckpt(...)`
Loads a saved Faster R-CNN checkpoint.

### `compute_image_ious_per_class(...)`
Computes IoU values grouped by class.

### `eval_miou_frcnn(...)`
Evaluates mean IoU for Faster R-CNN outputs.

### `eval_test_frcnn_miou_and_time(...)`
Measures mIoU and average inference time on the test set.

### `plot_cm(...)`
Plots the confusion matrix.

### `pr_and_cm_from_detections(...)`
Computes precision/recall-style counts and confusion matrix values from detections.

### `eval_frcnn_on_test(...)`
Runs final Faster R-CNN evaluation on the test set.

### `top_confusions(...)`
Shows the most common class confusions.

## When to use this file
Use this notebook when you want to:
- train the Faster R-CNN baseline
- analyze class confusion and IoU
- compare Faster R-CNN performance against YOLO or Cascade

---

# 3) `AI_TECH_PROJECT_TRAIN_casecade_frcnn_v2.ipynb`

## Purpose
This notebook is used to train the custom **Cascade Faster R-CNN** style detector in **Google Colab**.

## Main job of this notebook
It performs:
- dataset copy/unzip from Drive
- train/validation split
- custom dataset creation
- custom cascade-style RoI head definition
- training loop
- checkpoint saving/resume
- extra evaluation for mIoU, timing, confusion matrix, and COCO metrics

## Main sections inside the notebook

### Setup
- mounts Drive
- installs `pycocotools`, `scikit-learn`, `tqdm`
- imports PyTorch/TorchVision tools

### Data split
- finds the COCO JSON file
- limits target dataset size
- writes train/validation annotation files

### Model
- defines helper functions for box encoding/decoding
- defines custom cascade-style heads
- builds the detector from a Faster R-CNN base

### Training
- training loop
- checkpoint save/load
- resume from checkpoint
- export/save final model

### Test + extra checks
- evaluate on test split
- compute mIoU
- measure inference time
- build confusion matrix
- run COCO evaluation

## Important functions

### `prepare_local_data()`
Copies dataset zip files from Drive to Colab local storage and extracts them.

### `find_coco_json(root_dir)`
Finds the COCO annotation JSON file inside the extracted dataset.

### `collate_batch(batch)`
Batch collation helper for detection data.

### `EggCocoDataset`
Dataset class for loading egg images and annotations.

### `encode_boxes(...)`
Encodes ground-truth boxes relative to proposal boxes.

### `decode_boxes(...)`
Decodes predicted box deltas back to image coordinates.

### `TwoFCHead`
Two fully connected layers used in the custom RoI head.

### `ClassAgnosticPredictor`
Predictor head with class scores and class-agnostic box regression.

### `subsample_labels(...)`
Samples training labels for the detector head.

### `fastrcnn_loss_agnostic(...)`
Computes classification and regression loss for the class-agnostic head.

### `CascadeRoIHeads`
Main custom cascade-style multi-stage RoI head.

### `coco_evaluate(...)`
Validation evaluation using COCO metrics.

### `train_one_epoch(...)`
Runs one training epoch for the Cascade Faster R-CNN model.

### `save_ckpt(...)`
Saves a checkpoint to disk.

### `try_resume(...)`
Loads the latest checkpoint and resumes training if available.

### `save_model_to_drive(...)`
Copies the final saved model/checkpoint back to Google Drive.

### `timed_frcnn_predict(...)`
Measures inference time for the detector.

### `compute_per_class_ious(...)`
Computes IoU statistics per class.

### `eval_test_frcnn_miou_and_time(...)`
Runs test-set mIoU + inference timing evaluation.

### `plot_cm(...)`
Plots confusion matrix.

### `pr_and_cm_from_detections(...)`
Builds precision/recall and confusion matrix counts from detections.

### `coco_eval_from_results(...)`
Runs COCO evaluation from saved detection results.

### `eval_frcnn_on_test(...)`
Runs final evaluation on the test set.

## When to use this file
Use this notebook when you want to:
- train the custom cascade model
- test a cascade-style architecture on your dataset
- compare against Faster R-CNN and YOLO

---

# 4) `app_v2.py`

## Purpose
This file is a **desktop GUI application** for reviewing prediction results image by image.

## Platform
Run this in:
- **VS Code**
- terminal
- local Python environment with PyQt5 installed

## Main job of this file
It creates a desktop window where you can:
- choose model weights
- choose an image folder
- choose a COCO ground-truth JSON file
- run prediction on all images
- compare prediction vs ground truth
- browse through images
- filter wrong predictions only
- export CSV
- save fail cases
- show a confusion matrix popup

## Important functions and classes

### `to_qimage(image_bgr)`
Converts an OpenCV BGR image into a Qt image object for display.

### `draw_detection_boxes(image_bgr, boxes_xyxy, labels, scores, label_map, color=...)`
Draws bounding boxes and class labels on the image.

### `compare_prediction_with_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.25)`
Matches predictions with ground truth using IoU and returns TP, FP, FN, and match details.

### `ImagePopup`
Simple popup dialog used to display a large image, such as the confusion matrix.

### `DetectionReviewApp`
Main GUI class that controls the whole application.

Inside this class:

#### `_build_ui()`
Creates all buttons, labels, preview areas, and layout.

#### `_connect_signals()`
Connects button clicks to the corresponding methods.

#### `refresh_path_info()`
Updates the path summary shown in the GUI.

#### `pick_model()`
Lets the user select a weight file.

#### `pick_image_folder()`
Lets the user choose the image directory.

#### `pick_gt_json()`
Lets the user choose the COCO JSON file.

#### `load_resources()`
Loads the model, COCO annotations, category mapping, and filename-to-image-id map.

#### `get_gt_for_image(file_name)`
Returns ground-truth boxes and labels for one image.

#### `build_detail_text(record)`
Creates the detailed text summary for one image.

#### `run_all_predictions()`
Runs the full prediction loop over all images and computes summary statistics.

#### `_run_single_image(image_path, file_name)`
Runs inference and comparison for one image.

#### `refresh_preview()`
Updates the displayed image pair in the GUI.

#### `show_prev()` / `show_next()`
Moves between images.

#### `toggle_wrong_filter()`
Shows only wrong images when enabled.

#### `export_csv()`
Saves a CSV summary of all image-level results.

#### `save_fail_cases()`
Writes only wrong examples into a `fail_cases/` folder.

#### `show_confusion_matrix()`
Builds and displays a confusion matrix in a popup.

### `main()`
Starts the Qt application.

## Important note about this app
Even though the UI text mentions **Cascade FRCNN**, the actual model loading inside the file uses `YOLO(self.model_path)`.  
So this GUI currently behaves like a **YOLO review app** unless you modify the loading logic.

## When to use this file
Use this file when you want a visual local inspection tool instead of command-line output.

---

# 5) `eval_all_v2.py`

## Purpose
This file is the **main command-line evaluation launcher**.

## Platform
Run in:
- **VS Code terminal**
- Command Prompt / PowerShell
- macOS Terminal / Linux Terminal

## Main job of this file
It:
- reads command-line arguments
- selects which model(s) to evaluate
- rebuilds Faster R-CNN / Cascade model architecture if needed
- loads model wrappers
- calls `run_full_evaluation(...)`
- writes outputs into `runs_eval/<run_name>/...`
- prints final summary numbers

## Main functions

### `count_lines_in_class_file(path)`
Counts non-empty lines in `classes.txt`.  
This is used to infer the number of object classes.

### `parse_args()`
Defines all command-line arguments such as:
- `--coco_json`
- `--img_dir`
- `--yolo8_best`
- `--yolo11_best`
- `--frcnn_best`
- `--cascade_best`
- `--classes_txt`
- `--num_classes`
- `--score_thr`
- `--iou_thr`
- `--device`
- `--run_name`
- `--limit_images`

### `get_torchvision_num_classes(args)`
Returns the number of classes needed to rebuild TorchVision detectors.

### `main()`
Main entry point:
- prepares output folder
- launches selected evaluations
- prints final summary metrics

## Supported models
- YOLOv8s
- YOLO11s
- Faster R-CNN
- Cascade Faster R-CNN

## Example commands

### Evaluate YOLOv8
```bash
python eval_all_v2.py \
  --coco_json test.json \
  --img_dir images \
  --yolo8_best best.pt \
  --device cuda
```

### Evaluate YOLO11
```bash
python eval_all_v2.py \
  --coco_json test.json \
  --img_dir images \
  --yolo11_best best.pt \
  --device cuda
```

### Evaluate Faster R-CNN
```bash
python eval_all_v2.py \
  --coco_json test.json \
  --img_dir images \
  --frcnn_best faster_rcnn_best.pth \
  --classes_txt classes.txt \
  --device cuda
```

### Evaluate Cascade Faster R-CNN
```bash
python eval_all_v2.py \
  --coco_json test.json \
  --img_dir images \
  --cascade_best cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --device cuda
```

### Debug using part of the dataset
```bash
python eval_all_v2.py \
  --coco_json test.json \
  --img_dir images \
  --cascade_best cascade_frcnn_best.pth \
  --classes_txt classes.txt \
  --limit_images 100
```

---

# 6) `evaluator_v2.py`

## Purpose
This file is the **evaluation backend** used by `eval_all_v2.py`.

## Main job of this file
It handles the core evaluation logic:
- load COCO data
- load ground truth per image
- store predictions in a shared format
- match predictions to ground truth
- build confusion matrix
- compute precision, recall, F1, and mean IoU
- compute COCO mAP using saved detections
- save CSV/JSON/PNG outputs

## Main data classes

### `Prediction`
Stores:
- `boxes`
- `scores`
- `labels`

### `GroundTruth`
Stores:
- `boxes`
- `labels`

## Important functions

### `load_coco_info(coco_json)`
Loads COCO annotations, category mappings, and image IDs.

### `get_image_path(coco, image_id, image_dir)`
Builds the full image path from a COCO image entry.

### `load_gt_for_image(coco, image_id, catid_to_label)`
Loads ground-truth boxes and labels for one image.

### `greedy_match(pred, gt, iou_thr=0.50)`
Greedily matches predictions to ground truth using IoU and prediction score order.

### `build_confusion_matrix(...)`
Builds the raw detection confusion matrix including background row/column.

### `save_confusion_plot(cm, labels, out_png, title)`
Saves the confusion matrix figure.

### `compute_class_metrics(...)`
Computes per-class TP, FP, FN, precision, recall, F1, and mean IoU.

### `coco_map_from_json(coco_gt, det_json_path)`
Runs official COCO evaluation from saved detection JSON.

### `save_predictions_as_coco_json(...)`
Writes model predictions to COCO detection JSON format.

### `run_full_evaluation(...)`
Main evaluation pipeline:
- inference on all images
- save detections JSON
- compute COCO mAP
- compute confusion matrix
- compute per-class metrics
- save summary files
- return summary dictionary

## Output files created by this module
Inside the output folder, it can generate:
- `detections_coco.json`
- `confusion_matrix_counts.csv`
- `confusion_matrix.png`
- `per_class_metrics.csv`
- `metrics_summary.json`

---

# 7) `models_v2.py`

## Purpose
This file defines the detector-building and model-loading logic.

## Main job of this file
It provides:
- a YOLO prediction wrapper
- a Faster R-CNN builder
- a custom Cascade Faster R-CNN builder
- a wrapper that loads saved TorchVision checkpoints and runs inference

## Important classes and functions

### `YOLOPredictor`
Loads a YOLO checkpoint and exposes `predict_one(...)`.

### `build_faster_rcnn(num_classes, min_size=600, max_size=1000, pretrained=False)`
Builds a Faster R-CNN model and replaces the box predictor head.

### `ClassAgnosticFastRCNNPredictor`
Small class for class score prediction + class-agnostic box regression.

### `decode_boxes(...)`
Decodes box deltas into image-space boxes.

### `refine_boxes(...)`
Applies decoded boxes and clips them to image size.

### `TwoFCHead`
Two fully connected layers used in the detector head.

### `ClassAgnosticPredictor`
Predictor for class score and class-agnostic box regression.

### `CascadeRoIHeads`
Custom multi-stage RoI head for the cascade-style detector.

### `build_cascade_frcnn(num_classes, min_size=600, max_size=1000)`
Creates the custom Cascade Faster R-CNN model starting from a Faster R-CNN backbone.

### `DetectorWrapper`
Loads a checkpoint, moves model to device, runs inference, and returns predictions in shared format.

## When to use this file
Use this module whenever you need to:
- rebuild a detector before loading `.pth`
- run TorchVision detector inference from saved checkpoints
- share one prediction format across different detector types

---

# 8) `requirements.txt`

## Purpose
This file lists the Python dependencies needed to run the project locally.

## Main packages included
- numerical/data handling: `numpy`, `pandas`
- image and plotting: `opencv-python`, `matplotlib`, `Pillow`, `seaborn`
- ML/CV: `torch`, `torchvision`, `ultralytics`, `scikit-learn`
- annotation/evaluation: `pycocotools`
- GUI: `PyQt5`
- utility/config: `tqdm`, `PyYAML`

## When to use this file
Use it right after creating the virtual environment:

```bash
pip install -r requirements.txt
```

---

## How to run the project locally after installation

### Run the desktop GUI
```bash
python app_v2.py
```

### Run command-line evaluation
```bash
python eval_all_v2.py --coco_json test.json --img_dir images --yolo8_best best.pt
```

---

## Suggested folder layout for another machine

```bash
AI_TECH_PROJECT/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── test.json
├── classes.txt
├── best.pt
├── faster_rcnn_best.pth
├── cascade_frcnn_best.pth
├── app_v2.py
├── eval_all_v2.py
├── evaluator_v2.py
├── models_v2.py
├── requirements.txt
└── README.md
```

---

## Common problems and how to fix them

### 1. `ModuleNotFoundError`
Cause:
- file names and import names do not match

Fix:
- rename the files or edit the imports

### 2. `pycocotools` install problem on Windows
Fix:
- try installing with pip inside the virtual environment first
- if it fails, use a compatible wheel or Conda-based install

### 3. CUDA not detected
Fix:
- check NVIDIA driver
- check CUDA-compatible PyTorch install
- try `device="cpu"` first to confirm the code works

### 4. No predictions shown in app
Fix:
- check that the selected weight file is the correct model
- check that image filenames in JSON match real files
- check confidence threshold and category mapping

### 5. Wrong number of classes for Faster/Cascade
Fix:
- provide `classes.txt`
- or provide `--num_classes`
- make sure the count matches the training setup

---

## Suggested usage order

If you are using this project for the first time, this order is the easiest:

1. Train models in Colab with the notebooks  
2. Download trained weights  
3. Move weights and dataset to local machine  
4. Fix import names if needed  
5. Install requirements  
6. Test the GUI with `app_v2.py`  
7. Run full benchmark with `eval_all_v2.py`

---

## Final note

In short:

- use the **notebooks** for training in **Google Colab**
- use the **Python scripts** for local evaluation in **VS Code**
- keep all paths, weights, class counts, and annotation files consistent
- fix import names first before trying to run the local files
