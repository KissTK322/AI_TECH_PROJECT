import os, json, time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision.ops import box_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt


@dataclass
class Prediction:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class GroundTruth:
    boxes: np.ndarray
    labels: np.ndarray


def load_coco(coco_json: str):
    coco = COCO(coco_json)
    img_ids = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    catid_to_contig = {c["id"]: i + 1 for i, c in enumerate(cats_sorted)}
    contig_to_catid = {v: k for k, v in catid_to_contig.items()}
    class_names = {catid_to_contig[c["id"]]: c["name"] for c in cats_sorted}
    return coco, img_ids, catid_to_contig, contig_to_catid, class_names


def coco_img_path(coco: COCO, img_id: int, img_dir: str) -> str:
    info = coco.loadImgs([img_id])[0]
    return os.path.join(img_dir, info["file_name"])


def load_gt_for_image(coco: COCO, img_id: int, catid_to_contig: Dict[int, int]) -> GroundTruth:
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    boxes, labels = [], []
    for a in anns:
        if a.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(catid_to_contig[a["category_id"]])

    if not boxes:
        return GroundTruth(
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )
    return GroundTruth(
        boxes=np.array(boxes, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
    )


def match_detections(pred: Prediction, gt: GroundTruth, iou_thr: float = 0.50):
    if pred.boxes.shape[0] == 0 or gt.boxes.shape[0] == 0:
        return [], list(range(gt.boxes.shape[0])), list(range(pred.boxes.shape[0]))

    ious = box_iou(torch.from_numpy(gt.boxes), torch.from_numpy(pred.boxes)).numpy()
    order = np.argsort(-pred.scores)

    matched_gt, matched_pred = set(), set()
    matches = []
    for pj in order:
        gi = int(np.argmax(ious[:, pj]))
        best_iou = float(ious[gi, pj])
        if best_iou >= iou_thr and gi not in matched_gt:
            matches.append((gi, int(pj), best_iou))
            matched_gt.add(gi)
            matched_pred.add(int(pj))

    unmatched_gt = [i for i in range(gt.boxes.shape[0]) if i not in matched_gt]
    unmatched_pred = [j for j in range(pred.boxes.shape[0]) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def build_confusion_counts(all_preds, coco, img_ids, catid_to_contig, num_classes, iou_thr):
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for img_id in tqdm(img_ids, desc="Confusion", leave=False):
        gt = load_gt_for_image(coco, img_id, catid_to_contig)
        pred = all_preds.get(
            img_id,
            Prediction(
                boxes=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            ),
        )

        matches, u_gt, u_pr = match_detections(pred, gt, iou_thr=iou_thr)

        for gi, pj, _ in matches:
            gt_c = int(gt.labels[gi]) - 1
            pr_c = int(pred.labels[pj]) - 1
            cm[gt_c, pr_c] += 1

        for gi in u_gt:
            gt_c = int(gt.labels[gi]) - 1
            cm[gt_c, bg] += 1

        for pj in u_pr:
            pr_c = int(pred.labels[pj]) - 1
            cm[bg, pr_c] += 1

    return cm


def plot_confusion_matrix(cm, labels, out_png, title):
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=60, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() * 0.60 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v == 0:
                continue
            plt.text(
                j, i, str(v),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.ylabel("GT")
    plt.xlabel("Pred")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def compute_pr_f1_miou_from_matches(all_preds, coco, img_ids, catid_to_contig, num_classes, iou_thr):
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)
    iou_sum = np.zeros((num_classes,), dtype=np.float64)
    iou_cnt = np.zeros((num_classes,), dtype=np.int64)

    for img_id in tqdm(img_ids, desc="P/R/F1", leave=False):
        gt = load_gt_for_image(coco, img_id, catid_to_contig)
        pred = all_preds.get(
            img_id,
            Prediction(
                boxes=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            ),
        )

        matches, u_gt, u_pr = match_detections(pred, gt, iou_thr=iou_thr)

        for gi, pj, iou in matches:
            gt_c = int(gt.labels[gi]) - 1
            pr_c = int(pred.labels[pj]) - 1
            if gt_c == pr_c:
                tp[gt_c] += 1
                iou_sum[gt_c] += float(iou)
                iou_cnt[gt_c] += 1
            else:
                fp[pr_c] += 1
                fn[gt_c] += 1

        for gi in u_gt:
            gt_c = int(gt.labels[gi]) - 1
            fn[gt_c] += 1

        for pj in u_pr:
            pr_c = int(pred.labels[pj]) - 1
            fp[pr_c] += 1

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    miou_cls = iou_sum / np.maximum(iou_cnt, 1)
    miou = float(miou_cls.mean()) if num_classes > 0 else 0.0

    return tp, fp, fn, precision, recall, f1, miou_cls, miou


import json
from pycocotools.cocoeval import COCOeval

def coco_map_from_detections_json(coco_gt, det_json_path: str):
    with open(det_json_path, "r", encoding="utf-8") as f:
        dets = json.load(f)
    if not dets:
        return {k: 0.0 for k in [
            "mAP_50_95","mAP_50","mAP_75","mAP_small","mAP_medium","mAP_large",
            "AR_1","AR_10","AR_100","AR_small","AR_medium","AR_large"
        ]}
    coco_dt = coco_gt.loadRes(dets)

    coco_dt = coco_gt.loadRes(dets)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    s = coco_eval.stats
    return {
        "mAP_50_95": float(s[0]),
        "mAP_50": float(s[1]),
        "mAP_75": float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
        "AR_1": float(s[6]),
        "AR_10": float(s[7]),
        "AR_100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }
    


def save_detections_coco_json(all_preds, img_ids, contig_to_catid, out_json):
    dets = []
    for img_id in img_ids:
        pred = all_preds.get(img_id, None)
        if pred is None or pred.boxes.shape[0] == 0:
            continue
        for box, score, lab in zip(pred.boxes, pred.scores, pred.labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            dets.append({
                "image_id": int(img_id),
                "category_id": int(contig_to_catid[int(lab)]),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dets, f)


def run_full_evaluation(model_name, coco_json, img_dir, predictor_fn, out_dir, score_thr=0.25, iou_thr_cm=0.50, limit_images=None):
    os.makedirs(out_dir, exist_ok=True)

    coco, img_ids, catid_to_contig, contig_to_catid, class_names = load_coco(coco_json)

    if limit_images is not None:
        img_ids = img_ids[:int(limit_images)]

    K = len(class_names)

    all_preds = {}
    t0 = time.time()
    for img_id in tqdm(img_ids, desc=f"Infer [{model_name}]"):
        img_path = coco_img_path(coco, img_id, img_dir)
        all_preds[img_id] = predictor_fn(img_path, score_thr=score_thr)
    infer_s = time.time() - t0

    det_json = os.path.join(out_dir, "detections_coco.json")
    save_detections_coco_json(all_preds, img_ids, contig_to_catid, det_json)
    map_stats = coco_map_from_detections_json(coco, det_json)

    tp, fp, fn, precision, recall, f1, miou_cls, miou = compute_pr_f1_miou_from_matches(
        all_preds, coco, img_ids, catid_to_contig, K, iou_thr=iou_thr_cm
    )

    cm = build_confusion_counts(all_preds, coco, img_ids, catid_to_contig, K, iou_thr=iou_thr_cm)
    labels = [class_names[i + 1] for i in range(K)] + ["__background__"]

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"))
    plot_confusion_matrix(cm, labels, os.path.join(out_dir, "confusion_matrix.png"), f"{model_name} Confusion Matrix (raw counts)")

    df = pd.DataFrame({
        "class": [class_names[i + 1] for i in range(K)],
        "tp": tp, "fp": fp, "fn": fn,
        "precision@iou": precision,
        "recall@iou": recall,
        "f1@iou": f1,
        "mean_iou_tp": miou_cls,
    })
    df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    summary = {
        "model": model_name,
        "num_images": int(len(img_ids)),
        "num_classes": int(K),
        "score_thr": float(score_thr),
        "iou_thr_for_pr_cm": float(iou_thr_cm),
        "inference_seconds_total": float(infer_s),
        "inference_ms_per_image": float(1000.0 * infer_s / max(len(img_ids), 1)),
        "coco_map": map_stats,
        "miou_tp_mean": float(miou),
        "macro_precision@iou": float(np.mean(precision)),
        "macro_recall@iou": float(np.mean(recall)),
        "macro_f1@iou": float(np.mean(f1)),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
