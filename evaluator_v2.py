import json
import os
import time
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm


@dataclass
class Prediction:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class GroundTruth:
    boxes: np.ndarray
    labels: np.ndarray


def load_coco_info(coco_json: str):
    coco = COCO(coco_json)
    image_ids = coco.getImgIds()

    categories = coco.loadCats(coco.getCatIds())
    categories = sorted(categories, key=lambda item: item["id"])

    catid_to_label = {item["id"]: idx + 1 for idx, item in enumerate(categories)}
    label_to_catid = {label: cat_id for cat_id, label in catid_to_label.items()}
    class_names = {catid_to_label[item["id"]]: item["name"] for item in categories}

    return coco, image_ids, catid_to_label, label_to_catid, class_names


def get_image_path(coco: COCO, image_id: int, image_dir: str) -> str:
    image_info = coco.loadImgs([image_id])[0]
    return os.path.join(image_dir, image_info["file_name"])


def load_gt_for_image(coco: COCO, image_id: int, catid_to_label: Dict[int, int]) -> GroundTruth:
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    boxes = []
    labels = []

    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        boxes.append([x, y, x + w, y + h])
        labels.append(catid_to_label[ann["category_id"]])

    if not boxes:
        return GroundTruth(
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )

    return GroundTruth(
        boxes=np.array(boxes, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
    )


def greedy_match(pred: Prediction, gt: GroundTruth, iou_thr: float = 0.50):
    if pred.boxes.shape[0] == 0 or gt.boxes.shape[0] == 0:
        return [], list(range(gt.boxes.shape[0])), list(range(pred.boxes.shape[0]))

    iou_table = box_iou(torch.from_numpy(gt.boxes), torch.from_numpy(pred.boxes)).numpy()
    pred_order = np.argsort(-pred.scores)

    used_gt = set()
    used_pred = set()
    matches = []

    for pred_idx in pred_order:
        gt_idx = int(np.argmax(iou_table[:, pred_idx]))
        best_iou = float(iou_table[gt_idx, pred_idx])

        if best_iou >= iou_thr and gt_idx not in used_gt:
            matches.append((gt_idx, int(pred_idx), best_iou))
            used_gt.add(gt_idx)
            used_pred.add(int(pred_idx))

    unmatched_gt = [idx for idx in range(gt.boxes.shape[0]) if idx not in used_gt]
    unmatched_pred = [idx for idx in range(pred.boxes.shape[0]) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def build_confusion_matrix(all_predictions, coco, image_ids, catid_to_label, num_classes, iou_thr):
    bg_index = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for image_id in tqdm(image_ids, desc="Confusion", leave=False):
        gt = load_gt_for_image(coco, image_id, catid_to_label)
        pred = all_predictions.get(
            image_id,
            Prediction(
                boxes=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            ),
        )

        matches, missed_gt, extra_pred = greedy_match(pred, gt, iou_thr=iou_thr)

        for gt_idx, pred_idx, _ in matches:
            gt_class = int(gt.labels[gt_idx]) - 1
            pred_class = int(pred.labels[pred_idx]) - 1
            cm[gt_class, pred_class] += 1

        for gt_idx in missed_gt:
            gt_class = int(gt.labels[gt_idx]) - 1
            cm[gt_class, bg_index] += 1

        for pred_idx in extra_pred:
            pred_class = int(pred.labels[pred_idx]) - 1
            cm[bg_index, pred_class] += 1

    return cm


def save_confusion_plot(cm, labels, out_png, title):
    plt.figure(figsize=(12, 10))
    image = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(image, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(labels))
    plt.xticks(tick_positions, labels, rotation=60, ha="right")
    plt.yticks(tick_positions, labels)

    threshold = cm.max() * 0.60 if cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            if value == 0:
                continue
            plt.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.ylabel("GT")
    plt.xlabel("Pred")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def compute_class_metrics(all_predictions, coco, image_ids, catid_to_label, num_classes, iou_thr):
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)
    iou_sum = np.zeros((num_classes,), dtype=np.float64)
    iou_count = np.zeros((num_classes,), dtype=np.int64)

    for image_id in tqdm(image_ids, desc="P/R/F1", leave=False):
        gt = load_gt_for_image(coco, image_id, catid_to_label)
        pred = all_predictions.get(
            image_id,
            Prediction(
                boxes=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            ),
        )

        matches, missed_gt, extra_pred = greedy_match(pred, gt, iou_thr=iou_thr)

        for gt_idx, pred_idx, iou in matches:
            gt_class = int(gt.labels[gt_idx]) - 1
            pred_class = int(pred.labels[pred_idx]) - 1

            if gt_class == pred_class:
                tp[gt_class] += 1
                iou_sum[gt_class] += float(iou)
                iou_count[gt_class] += 1
            else:
                fp[pred_class] += 1
                fn[gt_class] += 1

        for gt_idx in missed_gt:
            gt_class = int(gt.labels[gt_idx]) - 1
            fn[gt_class] += 1

        for pred_idx in extra_pred:
            pred_class = int(pred.labels[pred_idx]) - 1
            fp[pred_class] += 1

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    mean_iou_per_class = iou_sum / np.maximum(iou_count, 1)
    mean_iou = float(mean_iou_per_class.mean()) if num_classes > 0 else 0.0

    return tp, fp, fn, precision, recall, f1, mean_iou_per_class, mean_iou


def coco_map_from_json(coco_gt, det_json_path: str):
    with open(det_json_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    if not detections:
        return {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def save_predictions_as_coco_json(all_predictions, image_ids, label_to_catid, out_json):
    detections = []

    for image_id in image_ids:
        pred = all_predictions.get(image_id)
        if pred is None or pred.boxes.shape[0] == 0:
            continue

        for box, score, label in zip(pred.boxes, pred.scores, pred.labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label_to_catid[int(label)]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detections, f)


def run_full_evaluation(
    model_name,
    coco_json,
    img_dir,
    predictor_fn,
    out_dir,
    score_thr=0.25,
    iou_thr_cm=0.50,
    limit_images=None,
):
    os.makedirs(out_dir, exist_ok=True)

    coco, image_ids, catid_to_label, label_to_catid, class_names = load_coco_info(coco_json)
    if limit_images is not None:
        image_ids = image_ids[: int(limit_images)]

    num_classes = len(class_names)
    all_predictions = {}

    start_time = time.time()
    for image_id in tqdm(image_ids, desc=f"Infer [{model_name}]"):
        img_path = get_image_path(coco, image_id, img_dir)
        all_predictions[image_id] = predictor_fn(img_path, score_thr=score_thr)
    total_infer_time = time.time() - start_time

    det_json_path = os.path.join(out_dir, "detections_coco.json")
    save_predictions_as_coco_json(all_predictions, image_ids, label_to_catid, det_json_path)
    map_stats = coco_map_from_json(coco, det_json_path)

    tp, fp, fn, precision, recall, f1, mean_iou_per_class, mean_iou = compute_class_metrics(
        all_predictions,
        coco,
        image_ids,
        catid_to_label,
        num_classes,
        iou_thr=iou_thr_cm,
    )

    cm = build_confusion_matrix(
        all_predictions,
        coco,
        image_ids,
        catid_to_label,
        num_classes,
        iou_thr=iou_thr_cm,
    )
    labels = [class_names[idx + 1] for idx in range(num_classes)] + ["__background__"]

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(out_dir, "confusion_matrix_counts.csv")
    )
    save_confusion_plot(
        cm,
        labels,
        os.path.join(out_dir, "confusion_matrix.png"),
        f"{model_name} Confusion Matrix (raw counts)",
    )

    per_class_df = pd.DataFrame(
        {
            "class": [class_names[idx + 1] for idx in range(num_classes)],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision@iou": precision,
            "recall@iou": recall,
            "f1@iou": f1,
            "mean_iou_tp": mean_iou_per_class,
        }
    )
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    summary = {
        "model": model_name,
        "num_images": int(len(image_ids)),
        "num_classes": int(num_classes),
        "score_thr": float(score_thr),
        "iou_thr_for_pr_cm": float(iou_thr_cm),
        "inference_seconds_total": float(total_infer_time),
        "inference_ms_per_image": float(1000.0 * total_infer_time / max(len(image_ids), 1)),
        "coco_map": map_stats,
        "miou_tp_mean": float(mean_iou),
        "macro_precision@iou": float(np.mean(precision)),
        "macro_recall@iou": float(np.mean(recall)),
        "macro_f1@iou": float(np.mean(f1)),
    }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
