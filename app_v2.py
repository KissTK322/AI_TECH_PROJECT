import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from torchvision.ops import box_iou
from ultralytics import YOLO


CONF_TH = 0.25
IOU_TH = 0.25


def to_qimage(image_bgr: np.ndarray) -> QImage:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape
    bytes_per_line = c * w
    return QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def draw_detection_boxes(image_bgr, boxes_xyxy, labels, scores, label_map, color=(0, 255, 0)):
    canvas = image_bgr.copy()

    for (x1, y1, x2, y2), label_id, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label_name = label_map.get(int(label_id), str(label_id))
        tag = f"{label_name} {float(score):.2f}"
        (text_w, text_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        top = max(0, y1 - text_h - 8)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            canvas,
            tag,
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return canvas


def compare_prediction_with_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.25):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, []

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    iou_table = box_iou(pred_tensor, gt_tensor).numpy()

    used_gt = set()
    matches = []
    true_positive = 0
    false_positive = 0

    candidate_order = np.argsort(-np.max(iou_table, axis=1))
    for pred_idx in candidate_order:
        gt_idx = int(np.argmax(iou_table[pred_idx]))
        best_iou = float(iou_table[pred_idx, gt_idx])

        if best_iou >= iou_thr and gt_idx not in used_gt:
            used_gt.add(gt_idx)
            class_ok = int(pred_labels[pred_idx]) == int(gt_labels[gt_idx])
            matches.append((pred_idx, gt_idx, best_iou, class_ok))
            if class_ok:
                true_positive += 1
            else:
                false_positive += 1
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - true_positive
    return true_positive, false_positive, false_negative, matches


class ImagePopup(QDialog):
    def __init__(self, title: str, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(950, 850)

        layout = QVBoxLayout()
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap)

        layout.addWidget(image_label)
        self.setLayout(layout)


class DetectionReviewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cascade FRCNN Review App")
        self.resize(1500, 900)

        self.model_path = ""
        self.image_dir = ""
        self.gt_json_path = ""

        self.model = None
        self.coco = None
        self.catid_to_name = {}
        self.model_cls_to_catid = {}
        self.filename_to_img_id = {}

        self.records = {}
        self.all_image_names = []
        self.wrong_image_names = []
        self.show_wrong_only = False
        self.current_index = 0
        self.summary = {}

        self._build_ui()
        self._connect_signals()
        self.refresh_path_info()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        top_bar = QHBoxLayout()
        self.btn_model = QPushButton("Choose Cascade FRCNN Weights")
        self.btn_images = QPushButton("Choose Image Folder")
        self.btn_gt = QPushButton("Choose GT COCO JSON")
        self.btn_run = QPushButton("Run All Predictions")

        for widget in [self.btn_model, self.btn_images, self.btn_gt, self.btn_run]:
            top_bar.addWidget(widget)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")

        preview_row = QHBoxLayout()
        self.pred_label = QLabel("Prediction")
        self.gt_label = QLabel("Ground Truth")

        for label in [self.pred_label, self.gt_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(650, 460)

        preview_row.addWidget(self.pred_label, 1)
        preview_row.addWidget(self.gt_label, 1)

        action_row = QHBoxLayout()
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        self.chk_wrong = QCheckBox("Show only wrong images")
        self.btn_export = QPushButton("Export CSV")
        self.btn_save_fail = QPushButton("Save fail cases")
        self.btn_confusion = QPushButton("Show confusion matrix")

        for widget in [
            self.btn_prev,
            self.btn_next,
            self.chk_wrong,
            self.btn_export,
            self.btn_save_fail,
            self.btn_confusion,
        ]:
            action_row.addWidget(widget)

        self.btn_confusion.setEnabled(False)

        self.summary_label = QLabel("SUMMARY: run evaluation first")
        self.summary_label.setWordWrap(True)

        self.info_label = QLabel("Pick a model, image folder, and GT json.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        page = QVBoxLayout()
        page.addLayout(top_bar)
        page.addWidget(self.progress)
        page.addLayout(preview_row)
        page.addLayout(action_row)
        page.addWidget(self.summary_label)
        page.addWidget(self.info_label)

        root.setLayout(page)

    def _connect_signals(self):
        self.btn_model.clicked.connect(self.pick_model)
        self.btn_images.clicked.connect(self.pick_image_folder)
        self.btn_gt.clicked.connect(self.pick_gt_json)
        self.btn_run.clicked.connect(self.run_all_predictions)

        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next.clicked.connect(self.show_next)
        self.chk_wrong.stateChanged.connect(self.toggle_wrong_filter)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_save_fail.clicked.connect(self.save_fail_cases)
        self.btn_confusion.clicked.connect(self.show_confusion_matrix)

    def show_message(self, text, title="Info"):
        QMessageBox.information(self, title, text)

    def refresh_path_info(self):
        self.info_label.setText(
            f"Model: {self.model_path or '(not selected)'}\n"
            f"Images: {self.image_dir or '(not selected)'}\n"
            f"GT: {self.gt_json_path or '(not selected)'}\n"
        )

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select casecade_frcnn_best.pt",
            "",
            "PyTorch weights (*.pt);;All files (*.*)",
        )
        if path:
            self.model_path = path
            self.refresh_path_info()

    def pick_image_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.image_dir = path
            self.refresh_path_info()

    def pick_gt_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COCO GT json",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if path:
            self.gt_json_path = path
            self.refresh_path_info()

    def load_resources(self):
        if not self.model_path or not Path(self.model_path).is_file():
            self.show_message("Please choose casecade_frcnn_best.pt", "Error")
            return False
        if not self.image_dir or not Path(self.image_dir).is_dir():
            self.show_message("Please choose image folder", "Error")
            return False
        if not self.gt_json_path or not Path(self.gt_json_path).is_file():
            self.show_message("Please choose GT COCO json", "Error")
            return False

        self.model = YOLO(self.model_path)
        self.coco = COCO(self.gt_json_path)

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.catid_to_name = {item["id"]: item["name"] for item in categories}

        categories_sorted = sorted(categories, key=lambda item: item["id"])
        self.model_cls_to_catid = {idx: item["id"] for idx, item in enumerate(categories_sorted)}

        self.filename_to_img_id = {
            item["file_name"]: item["id"] for item in self.coco.dataset["images"]
        }
        return True

    def get_gt_for_image(self, file_name):
        image_id = self.filename_to_img_id.get(file_name)
        if image_id is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def current_list(self):
        return self.wrong_image_names if self.show_wrong_only else self.all_image_names

    def build_detail_text(self, record):
        pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["pred_labels"]]
        gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["gt_labels"]]

        lines = [
            "Pred classes: " + (", ".join(pred_names) if pred_names else "(none)"),
            "GT classes: " + (", ".join(gt_names) if gt_names else "(none)"),
        ]

        if len(record["matches"]) == 0:
            if len(pred_names) == 0 and len(gt_names) == 0:
                lines.append("Per-object compare: no objects in either prediction or ground truth")
            else:
                lines.append("Per-object compare: no matched boxes")
            return "\n".join(lines)

        lines.append("Per-object compare:")

        used_pred = set()
        used_gt = set()

        for idx, (pred_i, gt_i, iou, ok_cls) in enumerate(record["matches"], start=1):
            used_pred.add(pred_i)
            used_gt.add(gt_i)

            pred_name = self.catid_to_name.get(int(record["pred_labels"][pred_i]), str(int(record["pred_labels"][pred_i])))
            gt_name = self.catid_to_name.get(int(record["gt_labels"][gt_i]), str(int(record["gt_labels"][gt_i])))
            status = "OK" if ok_cls else "WRONG CLASS"

            lines.append(
                f"  {idx}. Pred = {pred_name} | GT = {gt_name} | IoU = {iou:.2f} | {status}"
            )

        extra_preds = []
        for pred_i in range(len(record["pred_labels"])):
            if pred_i not in used_pred:
                extra_preds.append(
                    self.catid_to_name.get(int(record["pred_labels"][pred_i]), str(int(record["pred_labels"][pred_i])))
                )
        if extra_preds:
            lines.append("Unmatched predictions: " + ", ".join(extra_preds))

        missed_gt = []
        for gt_i in range(len(record["gt_labels"])):
            if gt_i not in used_gt:
                missed_gt.append(
                    self.catid_to_name.get(int(record["gt_labels"][gt_i]), str(int(record["gt_labels"][gt_i])))
                )
        if missed_gt:
            lines.append("Missed ground truths: " + ", ".join(missed_gt))

        return "\n".join(lines)

    def run_all_predictions(self):
        if not self.load_resources():
            return

        image_entries = self.coco.dataset["images"]
        total_images = len(image_entries)

        self.records = {}
        self.all_image_names = []
        self.wrong_image_names = []
        self.current_index = 0
        self.btn_confusion.setEnabled(False)

        self.progress.setRange(0, total_images)
        self.progress.setValue(0)

        obj_tp = obj_fp = obj_fn = 0
        img_tp = img_fp = img_fn = img_tn = 0

        buttons_to_disable = [
            self.btn_run,
            self.btn_prev,
            self.btn_next,
            self.btn_export,
            self.btn_save_fail,
            self.btn_confusion,
        ]
        for button in buttons_to_disable:
            button.setEnabled(False)

        for step, image_info in enumerate(image_entries, start=1):
            file_name = image_info["file_name"]
            image_path = str(Path(self.image_dir) / file_name)

            self.progress.setValue(step)
            self.progress.setFormat(f"Predicting {step}/{total_images}: {file_name}")
            QApplication.processEvents()

            record = self._run_single_image(image_path, file_name)
            self.records[file_name] = record

            obj_tp += record["tp"]
            obj_fp += record["fp"]
            obj_fn += record["fn"]

            has_gt = len(record["gt_labels"]) > 0
            has_pred = len(record["pred_labels"]) > 0

            if not has_gt and not has_pred:
                img_tn += 1
            elif not has_gt and has_pred:
                img_fp += 1
            elif has_gt and record["tp"] == 0:
                img_fn += 1
            else:
                img_tp += 1

        self.all_image_names = list(self.records.keys())
        self.wrong_image_names = [name for name, rec in self.records.items() if rec["wrong"]]

        num_correct = len(self.all_image_names) - len(self.wrong_image_names)
        img_acc = num_correct / max(len(self.all_image_names), 1)

        self.summary = {
            "TP": obj_tp,
            "FP": obj_fp,
            "FN": obj_fn,
            "TP_img": img_tp,
            "FP_img": img_fp,
            "FN_img": img_fn,
            "TN_img": img_tn,
            "wrong_images": len(self.wrong_image_names),
            "total_images": len(self.all_image_names),
            "image_accuracy": img_acc,
        }

        self.summary_label.setText(
            "SUMMARY\n"
            f"Object-level: TP={obj_tp}  FP={obj_fp}  FN={obj_fn}\n"
            f"Image-level: TP={img_tp}  FP={img_fp}  FN={img_fn}  TN={img_tn}\n"
            f"Wrong images={len(self.wrong_image_names)} / {len(self.all_image_names)}   "
            f"(image-acc={img_acc * 100:.2f}%)"
        )

        self.progress.setFormat("Done")

        for button in buttons_to_disable:
            button.setEnabled(True)
        self.btn_confusion.setEnabled(True)

        self.current_index = 0
        self.refresh_preview()
        self.show_message(
            f"Done. total={len(self.all_image_names)} | wrong={len(self.wrong_image_names)} | "
            f"image-acc={img_acc * 100:.2f}%",
            "Finished",
        )

    def _run_single_image(self, image_path, file_name):
        empty_record = {
            "file": image_path,
            "pred_boxes": np.zeros((0, 4), np.float32),
            "pred_scores": np.zeros((0,), np.float32),
            "pred_labels": np.zeros((0,), np.int64),
            "gt_boxes": np.zeros((0, 4), np.float32),
            "gt_labels": np.zeros((0,), np.int64),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "wrong": True,
            "matches": [],
        }

        if not Path(image_path).is_file():
            return empty_record

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return empty_record

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        prediction = self.model.predict(
            source=image_rgb,
            conf=CONF_TH,
            iou=IOU_TH,
            verbose=False,
        )[0]

        if prediction.boxes is None or len(prediction.boxes) == 0:
            pred_boxes = np.zeros((0, 4), np.float32)
            pred_scores = np.zeros((0,), np.float32)
            pred_labels = np.zeros((0,), np.int64)
        else:
            pred_boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_scores = prediction.boxes.conf.cpu().numpy().astype(np.float32)
            raw_labels = prediction.boxes.cls.cpu().numpy().astype(np.int64)
            pred_labels = np.array(
                [self.model_cls_to_catid[int(label)] for label in raw_labels],
                dtype=np.int64,
            )

        gt_boxes, gt_labels = self.get_gt_for_image(file_name)
        tp, fp, fn, matches = compare_prediction_with_gt(
            pred_boxes,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_thr=IOU_TH,
        )

        return {
            "file": image_path,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "wrong": (fp > 0) or (fn > 0),
            "matches": matches,
        }

    def refresh_preview(self):
        image_names = self.current_list()
        if not image_names:
            self.pred_label.setText("No images to show")
            self.gt_label.setText("No images to show")
            return

        file_name = image_names[self.current_index % len(image_names)]
        record = self.records[file_name]

        image_bgr = cv2.imread(record["file"])
        if image_bgr is None:
            self.info_label.setText(f"Cannot read: {record['file']}")
            return

        pred_vis = draw_detection_boxes(
            image_bgr,
            record["pred_boxes"],
            record["pred_labels"],
            record["pred_scores"],
            self.catid_to_name,
            color=(0, 255, 0),
        )
        gt_vis = draw_detection_boxes(
            image_bgr,
            record["gt_boxes"],
            record["gt_labels"],
            np.ones((len(record["gt_labels"]),), np.float32),
            self.catid_to_name,
            color=(0, 0, 255),
        )

        pred_pix = QPixmap.fromImage(to_qimage(pred_vis)).scaled(
            self.pred_label.width(),
            self.pred_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        gt_pix = QPixmap.fromImage(to_qimage(gt_vis)).scaled(
            self.gt_label.width(),
            self.gt_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.pred_label.setPixmap(pred_pix)
        self.gt_label.setPixmap(gt_pix)

        detail_text = self.build_detail_text(record)
        self.info_label.setText(
            f"{file_name}\n"
            f"TP={record['tp']} FP={record['fp']} FN={record['fn']} wrong={record['wrong']}  |  "
            f"conf={CONF_TH}, iou={IOU_TH}\n\n"
            f"{detail_text}\n\n"
            f"Model: {self.model_path}\n"
            f"Images: {self.image_dir}\n"
            f"GT: {self.gt_json_path}"
        )

    def show_prev(self):
        image_names = self.current_list()
        if not image_names:
            return
        self.current_index = (self.current_index - 1) % len(image_names)
        self.refresh_preview()

    def show_next(self):
        image_names = self.current_list()
        if not image_names:
            return
        self.current_index = (self.current_index + 1) % len(image_names)
        self.refresh_preview()

    def toggle_wrong_filter(self, _):
        self.show_wrong_only = self.chk_wrong.isChecked()
        self.current_index = 0
        self.refresh_preview()

    def export_csv(self):
        if not self.records:
            self.show_message("Run prediction first.", "Error")
            return

        default_path = str(Path(self.image_dir) / "summary_cascade_frcnn.csv")
        out_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_path, "CSV (*.csv)")
        if not out_path:
            return

        rows = []
        for file_name, record in self.records.items():
            pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["pred_labels"]]
            gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["gt_labels"]]

            rows.append(
                {
                    "file_name": file_name,
                    "tp": record["tp"],
                    "fp": record["fp"],
                    "fn": record["fn"],
                    "wrong": record["wrong"],
                    "num_pred": int(len(record["pred_labels"])),
                    "num_gt": int(len(record["gt_labels"])),
                    "pred_class_names": " | ".join(pred_names),
                    "gt_class_names": " | ".join(gt_names),
                    "detail": self.build_detail_text(record).replace("\n", " || "),
                }
            )

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.show_message(f"Saved: {out_path}", "CSV")

    def save_fail_cases(self):
        if not self.records:
            self.show_message("Run prediction first.", "Error")
            return

        out_dir = Path(self.image_dir) / "fail_cases"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for file_name, record in self.records.items():
            if not record["wrong"]:
                continue

            image_bgr = cv2.imread(record["file"])
            if image_bgr is None:
                continue

            merged_view = image_bgr.copy()
            merged_view = draw_detection_boxes(
                merged_view,
                record["gt_boxes"],
                record["gt_labels"],
                np.ones((len(record["gt_labels"]),), np.float32),
                self.catid_to_name,
                color=(0, 0, 255),
            )
            merged_view = draw_detection_boxes(
                merged_view,
                record["pred_boxes"],
                record["pred_labels"],
                record["pred_scores"],
                self.catid_to_name,
                color=(0, 255, 0),
            )

            save_path = out_dir / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), merged_view)
            saved_count += 1

        self.show_message(f"Saved {saved_count} images to: {out_dir}", "Done")

    def show_confusion_matrix(self):
        if not self.records or not self.coco:
            self.show_message("Run prediction first.", "Error")
            return

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories_sorted = sorted(categories, key=lambda item: item["id"])
        category_ids = [item["id"] for item in categories_sorted]
        category_names = [item["name"] for item in categories_sorted]

        bg_id = -1
        labels = category_ids + [bg_id]
        label_names = category_names + ["__background__"]
        index_map = {cat_id: idx for idx, cat_id in enumerate(labels)}

        size = len(labels)
        cm = np.zeros((size, size), dtype=np.int64)

        for record in self.records.values():
            pred_boxes = record["pred_boxes"]
            pred_labels = record["pred_labels"]
            gt_boxes = record["gt_boxes"]
            gt_labels = record["gt_labels"]

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                for pred_label in pred_labels:
                    if int(pred_label) in index_map:
                        cm[index_map[bg_id], index_map[int(pred_label)]] += 1
                continue

            if len(pred_boxes) == 0 and len(gt_boxes) > 0:
                for gt_label in gt_labels:
                    if int(gt_label) in index_map:
                        cm[index_map[int(gt_label)], index_map[bg_id]] += 1
                continue

            pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            iou_table = box_iou(pred_tensor, gt_tensor).numpy()

            used_gt = set()
            used_pred = set()

            candidate_order = np.argsort(-np.max(iou_table, axis=1))
            for pred_idx in candidate_order:
                gt_idx = int(np.argmax(iou_table[pred_idx]))
                best_iou = float(iou_table[pred_idx, gt_idx])

                if best_iou >= IOU_TH and gt_idx not in used_gt:
                    used_gt.add(gt_idx)
                    used_pred.add(pred_idx)

                    gt_label = int(gt_labels[gt_idx])
                    pred_label = int(pred_labels[pred_idx])
                    if gt_label in index_map and pred_label in index_map:
                        cm[index_map[gt_label], index_map[pred_label]] += 1

            for pred_idx in range(len(pred_boxes)):
                if pred_idx not in used_pred:
                    pred_label = int(pred_labels[pred_idx])
                    if pred_label in index_map:
                        cm[index_map[bg_id], index_map[pred_label]] += 1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in used_gt:
                    gt_label = int(gt_labels[gt_idx])
                    if gt_label in index_map:
                        cm[index_map[gt_label], index_map[bg_id]] += 1

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Detection Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(label_names, rotation=90, fontsize=8)
        ax.set_yticklabels(label_names, fontsize=8)

        for row in range(size):
            for col in range(size):
                value = cm[row, col]
                if value != 0:
                    ax.text(col, row, str(value), ha="center", va="center", fontsize=7)

        fig.tight_layout()
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)

        image_bgr = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        pixmap = QPixmap.fromImage(to_qimage(image_bgr)).scaled(
            900,
            800,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        dialog = ImagePopup("Confusion Matrix", pixmap, parent=self)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = DetectionReviewApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
