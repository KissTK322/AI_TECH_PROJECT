import numpy as np
import torch
import cv2

from ultralytics import YOLO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from evaluator import Prediction

class YOLOUltralyticsWrapper:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.model = YOLO(weights_path)
        self.device = device

    def predict_one(self, img_path: str, score_thr: float = 0.25) -> Prediction:
        res = self.model.predict(source=img_path, conf=score_thr, iou=0.7, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return Prediction(
                boxes=np.zeros((0,4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            )
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        labels = (res.boxes.cls.cpu().numpy().astype(np.int64) + 1)
        return Prediction(boxes=boxes, scores=scores, labels=labels)

def build_faster_rcnn(num_classes: int, min_size=600, max_size=1000, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=min_size, max_size=max_size
    )
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=min_size, max_size=max_size)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.batch_size_per_image = 128
    rpn = model.rpn
    if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
        rpn._pre_nms_top_n["training"] = 1000
        rpn._pre_nms_top_n["testing"]  = 1000
    if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
        rpn._post_nms_top_n["training"] = 500
        rpn._post_nms_top_n["testing"]  = 500
    return model

import torch
import torch.nn as nn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection._utils import BoxCoder
from torchvision.ops import boxes as box_ops


class ClassAgnosticFastRCNNPredictor(nn.Module):
    """
    Matches your checkpoint:
      - cls_score: (num_classes)
      - bbox_pred: (4) class-agnostic
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops

def decode_boxes(proposals, deltas, stds=(0.1,0.1,0.2,0.2), clip_val=math.log(1000.0/16)):
    px1, py1, px2, py2 = proposals.unbind(dim=1)
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    px = px1 + 0.5 * pw
    py = py1 + 0.5 * ph

    wx, wy, ww, wh = stds
    dx = deltas[:, 0] * wx
    dy = deltas[:, 1] * wy
    dw = (deltas[:, 2] * ww).clamp(max=clip_val)
    dh = (deltas[:, 3] * wh).clamp(max=clip_val)

    gx = dx * pw + px
    gy = dy * ph + py
    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    return torch.stack([x1, y1, x2, y2], dim=1)

def refine_boxes(proposals, deltas, stds, image_shape):
    boxes = decode_boxes(proposals, deltas, stds=stds)
    boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
    return boxes


class TwoFCHead(nn.Module):
    def __init__(self, in_channels, fc_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ClassAgnosticPredictor(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_dim, num_classes)  # include background=0
        self.bbox_pred = nn.Linear(in_dim, 4)            # class-agnostic

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class CascadeRoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        num_classes,
        feat_channels,
        num_stages=3,
        stage_bbox_stds=((0.1,0.1,0.2,0.2),(0.05,0.05,0.1,0.1),(0.033,0.033,0.067,0.067)),
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.stage_bbox_stds = stage_bbox_stds

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        out_size = self.box_roi_pool.output_size
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        pool_h, pool_w = out_size

        in_dim = feat_channels * pool_h * pool_w

        self.box_heads = nn.ModuleList([TwoFCHead(in_dim, 1024) for _ in range(num_stages)])
        self.box_predictors = nn.ModuleList([ClassAgnosticPredictor(1024, num_classes) for _ in range(num_stages)])

    def forward(self, features, proposals, image_shapes, targets=None):

        if targets is not None:
            return [], {}

        cur_props = proposals
        last_logits_split = None

        for s in range(self.num_stages):
            stds = self.stage_bbox_stds[s]

            box_feats = self.box_roi_pool(features, cur_props, image_shapes)
            box_feats = self.box_heads[s](box_feats)
            class_logits, box_regression = self.box_predictors[s](box_feats)

            counts = [p.shape[0] for p in cur_props]
            logits_split = torch.split(class_logits, counts, dim=0)
            deltas_split = torch.split(box_regression, counts, dim=0)

            next_props = []
            for props_i, deltas_i, img_shape in zip(cur_props, deltas_split, image_shapes):
                refined = refine_boxes(props_i, deltas_i, stds=stds, image_shape=img_shape)
                next_props.append(refined)

            cur_props = next_props
            last_logits_split = logits_split

        detections = []
        for boxes_i, logits_i, img_shape in zip(cur_props, last_logits_split, image_shapes):
            boxes_i = box_ops.clip_boxes_to_image(boxes_i, img_shape)
            probs = F.softmax(logits_i, dim=1)

            all_boxes, all_scores, all_labels = [], [], []
            for c in range(1, self.num_classes):
                sc = probs[:, c]
                keep = torch.nonzero(sc > self.score_thresh).squeeze(1)
                if keep.numel() == 0:
                    continue
                all_boxes.append(boxes_i[keep])
                all_scores.append(sc[keep])
                all_labels.append(torch.full((keep.numel(),), c, dtype=torch.int64, device=boxes_i.device))

            if len(all_boxes) == 0:
                detections.append({
                    "boxes": torch.zeros((0,4), device=boxes_i.device),
                    "scores": torch.zeros((0,), device=boxes_i.device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=boxes_i.device),
                })
                continue

            boxes_cat  = torch.cat(all_boxes,  dim=0)
            scores_cat = torch.cat(all_scores, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)

            keep_small = box_ops.remove_small_boxes(boxes_cat, min_size=1.0)
            boxes_cat  = boxes_cat[keep_small]
            scores_cat = scores_cat[keep_small]
            labels_cat = labels_cat[keep_small]

            keep_nms = box_ops.batched_nms(boxes_cat, scores_cat, labels_cat, self.nms_thresh)
            keep_nms = keep_nms[: self.detections_per_img]

            detections.append({
                "boxes":  boxes_cat[keep_nms],
                "scores": scores_cat[keep_nms],
                "labels": labels_cat[keep_nms],
            })

        return detections, {}


def build_cascade_frcnn(num_classes: int, min_size=600, max_size=1000):
    base = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=min_size, max_size=max_size
    )

    rpn = base.rpn
    if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
        rpn._pre_nms_top_n["training"] = 1000
        rpn._pre_nms_top_n["testing"]  = 1000
    if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
        rpn._post_nms_top_n["training"] = 500
        rpn._post_nms_top_n["testing"]  = 500

    feat_channels = getattr(base.backbone, "out_channels", 256)

    base.roi_heads = CascadeRoIHeads(
        box_roi_pool=base.roi_heads.box_roi_pool,
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_stages=3,
        stage_bbox_stds=((0.1,0.1,0.2,0.2),(0.05,0.05,0.1,0.1),(0.033,0.033,0.067,0.067)),
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )
    return base


# ---------- 5) wrapper that loads checkpoint dict ----------
class TorchVisionDetectorWrapper:
    def __init__(self, model, weights_state_dict_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        ck = torch.load(weights_state_dict_path, map_location="cpu")
        if isinstance(ck, dict) and "model_state_dict" in ck:
            sd = ck["model_state_dict"]
        else:
            sd = ck

        drop_prefix = ("roi_heads.box_head.", "roi_heads.box_predictor.")
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefix)}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Loaded weights.")
        print("Missing keys:", len(missing))
        if len(missing):
            print("\n--- Missing keys ---")
            for k in missing:
                print(k)
        print("\n--- Unexpected keys ---")
        for k in unexpected:
            print(k)

        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_one(self, img_path: str, score_thr: float = 0.25) -> Prediction:
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = img.to(self.device)

        out = self.model([img])[0]
        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"].detach().cpu().numpy().astype(np.int64)

        keep = scores >= score_thr
        return Prediction(boxes=boxes[keep], scores=scores[keep], labels=labels[keep])