# AI_TECH_PROJECT

This project focuses on parasitic egg detection from microscopic images using multiple object detection models:

- YOLOv8s
- YOLO11s
- Faster R-CNN
- Cascade Faster R-CNN

The repository includes training notebooks, command-line evaluation scripts, model-building code, and desktop GUI applications for prediction and result visualization.

---

## Project Structure

```bash
.
├── AI_TECH_PROJECT_TRAINING_YOLO.ipynb
├── AI_TECH_PROJECT_TRAINING_FASTER_RCNN.ipynb
├── AI_TECH_PROJECT_TRAIN_casecade_frcnn.ipynb
├── app.py
├── eval_all.py
├── evaluator.py
├── models.py
└── README.md
```

# Installation

Install casecade_frcnn.pt from google drive

```bash
https://drive.google.com/file/d/1cE0rBZpSlWlBBJfgi-74nrDaNjO01ycC/view?usp=sharing
```

Create and activate a virtual environment, then install all required packages.

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```
---

## `requirements.txt`

```txt
numpy
pandas
opencv-python
matplotlib
PyQt5
torch
torchvision
ultralytics
pycocotools
tqdm
```

## README section

````markdown
# Installation

Create and activate a virtual environment, then install all required packages.

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
````

---

# Main Files

## `app.py`

Desktop GUI application for running a YOLO model on an image folder and comparing predictions with COCO ground truth.
The application allows the user to choose a model file, an image folder, and a COCO JSON file, then run prediction on all images, browse results, export a CSV summary, save failed cases, and display a confusion matrix.  

## `eval_all.py`

Command-line script for evaluating one or more trained models on a COCO-format test set.
It supports YOLOv8s, YOLO11s, Faster R-CNN, and Cascade Faster R-CNN. The script saves outputs inside `runs_eval/<run_name>/` and prints a final summary including mAP50-95, mAP50, mean IoU, macro F1, and inference time per image. 

## `evaluator.py`

Core evaluation module used by the CLI pipeline.
It loads COCO annotations, matches predictions with ground truth, computes confusion matrix counts, calculates precision/recall/F1/mIoU, runs COCO mAP evaluation, and saves result files such as `confusion_matrix.png`, `per_class_metrics.csv`, and `metrics_summary.json`.  

## `models.py`

Model helper module for building and loading detectors.
It provides a YOLO wrapper, a Faster R-CNN builder, and a Cascade Faster R-CNN builder with custom cascade RoI heads and class-agnostic box regression.  

---

# How to Use `app.py`

## Purpose

`app.py` is a desktop GUI for evaluating a YOLO model on a folder of test images using COCO-format annotations. It shows predicted boxes and ground-truth boxes side by side, then lets the user inspect image-by-image results. 

## Required Inputs

Before running the app, prepare these files:

* a trained YOLO weight file (`.pt`)
* a folder containing test images
* a COCO-format ground-truth JSON file

The app checks that all three inputs exist before loading the model and annotations. It then loads the YOLO weights with `YOLO(self.model_path)` and loads the COCO file with `COCO(self.gt_json)`. 

## Run the App

```bash
python app.py
```

This launches the PyQt desktop interface. The program starts from `main()`, creates a Qt application, opens the `YOLOGUI` window, and runs the event loop. 

## GUI Workflow

### Step 1 — Choose Model

Click **Choose Cascade FRCNN Weights** and select your trained YOLO `.pt` file.
The selected path will appear in the information panel.  

### Step 2 — Choose Image Folder

Click **Choose Image Folder** and select the folder containing all test images. 

### Step 3 — Choose GT COCO JSON

Click **Choose GT COCO json** and select the annotation file in COCO format. 

### Step 4 — Run Prediction

Click **RUN Predict All** to evaluate the whole folder.
The app will:

* load the YOLO model
* load the COCO annotations
* map file names to image IDs
* run prediction on each image
* compare predictions with ground truth using IoU and class matching
* calculate TP, FP, and FN
* store all results for visualization and export 

## What the App Shows

After prediction, the GUI displays:

* **Predicted image** with predicted boxes
* **Ground-truth image** with GT boxes
* **Summary text**
* **Detailed per-image information** 

The app also supports:

* **Prev / Next** for image navigation
* **Show only wrong images**
* **Export CSV summary**
* **Save fail cases to `fail_cases/`**
* **Show Confusion Matrix**  

## Output Features

### Export CSV

The app can export a CSV summary of all evaluated images, including TP, FP, FN, wrong/correct status, predicted classes, ground-truth classes, and detail text. 

### Save Failed Cases

The app can save only incorrect cases into a `fail_cases/` folder for manual review. These saved images include both GT boxes and predicted boxes. 

### Show Confusion Matrix

The app can generate and display a confusion matrix directly inside the GUI. It includes all categories plus a background row/column to represent false positives and false negatives. 

---

# How to Use `eval_all.py`

## Purpose

`eval_all.py` is the main command-line evaluation script. It runs one or more models on a COCO-format dataset and writes all evaluation outputs into a `runs_eval/<run_name>/` folder. 

## Required Arguments

```bash
--coco_json
--img_dir
```

These two arguments are always required. The script also accepts model paths and evaluation settings such as score threshold, IoU threshold, device, run name, and optional image limit. 

## Supported Model Arguments

```bash
--yolo8_best
--yolo11_best
--frcnn_best
--cascade_best
```

For Faster R-CNN and Cascade Faster R-CNN, you must also provide either:

```bash
--classes_txt
```

or

```bash
--num_classes
```

because the script needs the number of classes to rebuild TorchVision-based models correctly. 

## Example: Evaluate YOLOv8s

```bash
python eval_all.py \
  --coco_json path/to/test.json \
  --img_dir path/to/images \
  --yolo8_best path/to/best.pt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_yolov8
```

## Example: Evaluate YOLO11s

```bash
python eval_all.py \
  --coco_json path/to/test.json \
  --img_dir path/to/images \
  --yolo11_best path/to/best.pt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_yolo11
```

## Example: Evaluate Faster R-CNN

```bash
python eval_all.py \
  --coco_json path/to/test.json \
  --img_dir path/to/images \
  --frcnn_best path/to/faster_rcnn_best.pth \
  --classes_txt path/to/classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_frcnn
```

## Example: Evaluate Cascade Faster R-CNN

```bash
python eval_all.py \
  --coco_json path/to/test.json \
  --img_dir path/to/images \
  --cascade_best path/to/cascade_frcnn_best.pth \
  --classes_txt path/to/classes.txt \
  --score_thr 0.25 \
  --iou_thr 0.50 \
  --device cuda \
  --run_name eval_cascade
```

## Optional Debug Mode

To evaluate only part of the dataset:

```bash
python eval_all.py \
  --coco_json path/to/test.json \
  --img_dir path/to/images \
  --cascade_best path/to/cascade_frcnn_best.pth \
  --classes_txt path/to/classes.txt \
  --limit_images 100
```

The script creates `runs_eval/<run_name>/`, evaluates the selected models, and prints a final summary for each one. 

---

# What `evaluator.py` Does

`evaluator.py` is the evaluation backend used by `eval_all.py`.
Its main responsibilities are:

* loading the COCO dataset
* converting COCO annotations into usable GT boxes and labels
* matching detections with GT using IoU
* building the confusion matrix
* computing precision, recall, F1, and mean IoU
* computing COCO mAP from saved detection JSON
* saving CSV and JSON summaries  

The main entry function is:

```python
run_full_evaluation(...)
```

This function runs inference for every image, writes `detections_coco.json`, computes COCO mAP, saves `confusion_matrix_counts.csv`, `confusion_matrix.png`, `per_class_metrics.csv`, and `metrics_summary.json`, then returns the final summary dictionary. 

---

# What `models.py` Does

`models.py` contains helper classes and builders for all supported models.

## YOLO Wrapper

`YOLOUltralyticsWrapper` loads a YOLO checkpoint and provides a `predict_one()` function that returns bounding boxes, scores, and labels in the shared `Prediction` format. 

## Faster R-CNN Builder

`build_faster_rcnn()` creates a `fasterrcnn_resnet50_fpn` model, replaces the default predictor head, and adjusts the RPN proposal limits. 

## Cascade Faster R-CNN Builder

`build_cascade_frcnn()` creates a custom Cascade Faster R-CNN using:

* multi-stage cascade refinement
* custom `TwoFCHead`
* class-agnostic predictor
* custom cascade RoI heads 

This file is used by `eval_all.py` to rebuild the correct model before loading checkpoint weights. 

---

# Output Files

After evaluation with `eval_all.py`, each model folder inside `runs_eval/<run_name>/` may contain:

* `detections_coco.json`
* `confusion_matrix_counts.csv`
* `confusion_matrix.png`
* `per_class_metrics.csv`
* `metrics_summary.json` 

These files can be used for analysis, visualization, and reporting.

---

# Notes

* `app.py` is for interactive desktop inspection of YOLO predictions. 
* `eval_all.py` is recommended for full benchmark evaluation across multiple models. 
* `evaluator.py` should usually be used indirectly through `eval_all.py`. 
* `models.py` should usually be imported by other scripts rather than run directly. 

