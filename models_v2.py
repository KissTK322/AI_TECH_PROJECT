import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops
from ultralytics import YOLO

from evaluator_student_style_v2 import Prediction


class YOLOPredictor:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.model = YOLO(weights_path)
        self.device = device

    def predict_one(self, img_path: str, score_thr: float = 0.25) -> Prediction:
        result = self.model.predict(source=img_path, conf=score_thr, iou=0.7, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return Prediction(
                boxes=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                labels=np.zeros((0,), np.int64),
            )

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        labels = result.boxes.cls.cpu().numpy().astype(np.int64) + 1
        return Prediction(boxes=boxes, scores=scores, labels=labels)


def build_faster_rcnn(num_classes: int, min_size=600, max_size=1000, pretrained=False):
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.batch_size_per_image = 128

    rpn = model.rpn
    if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
        rpn._pre_nms_top_n["training"] = 1000
        rpn._pre_nms_top_n["testing"] = 1000
    if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
        rpn._post_nms_top_n["training"] = 500
        rpn._post_nms_top_n["testing"] = 500

    return model


class ClassAgnosticFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        box_deltas = self.bbox_pred(x)
        return scores, box_deltas


def decode_boxes(proposals, deltas, stds=(0.1, 0.1, 0.2, 0.2), clip_val=math.log(1000.0 / 16)):
    px1, py1, px2, py2 = proposals.unbind(dim=1)

    proposal_w = (px2 - px1).clamp(min=1e-6)
    proposal_h = (py2 - py1).clamp(min=1e-6)
    proposal_cx = px1 + 0.5 * proposal_w
    proposal_cy = py1 + 0.5 * proposal_h

    wx, wy, ww, wh = stds
    dx = deltas[:, 0] * wx
    dy = deltas[:, 1] * wy
    dw = (deltas[:, 2] * ww).clamp(max=clip_val)
    dh = (deltas[:, 3] * wh).clamp(max=clip_val)

    pred_cx = dx * proposal_w + proposal_cx
    pred_cy = dy * proposal_h + proposal_cy
    pred_w = torch.exp(dw) * proposal_w
    pred_h = torch.exp(dh) * proposal_h

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def refine_boxes(proposals, deltas, stds, image_shape):
    refined = decode_boxes(proposals, deltas, stds=stds)
    refined = box_ops.clip_boxes_to_image(refined, image_shape)
    return refined


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
        self.cls_score = nn.Linear(in_dim, num_classes)
        self.bbox_pred = nn.Linear(in_dim, 4)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class CascadeRoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        num_classes,
        feat_channels,
        num_stages=3,
        stage_bbox_stds=((0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067)),
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
        self.box_predictors = nn.ModuleList(
            [ClassAgnosticPredictor(1024, num_classes) for _ in range(num_stages)]
        )

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            return [], {}

        current_props = proposals
        last_logits_split = None

        for stage_idx in range(self.num_stages):
            stds = self.stage_bbox_stds[stage_idx]

            pooled_features = self.box_roi_pool(features, current_props, image_shapes)
            pooled_features = self.box_heads[stage_idx](pooled_features)
            class_logits, box_regression = self.box_predictors[stage_idx](pooled_features)

            counts = [proposal.shape[0] for proposal in current_props]
            logits_split = torch.split(class_logits, counts, dim=0)
            deltas_split = torch.split(box_regression, counts, dim=0)

            next_props = []
            for proposal_set, delta_set, image_shape in zip(current_props, deltas_split, image_shapes):
                next_props.append(refine_boxes(proposal_set, delta_set, stds=stds, image_shape=image_shape))

            current_props = next_props
            last_logits_split = logits_split

        detections = []
        for boxes_per_image, logits_per_image, image_shape in zip(current_props, last_logits_split, image_shapes):
            boxes_per_image = box_ops.clip_boxes_to_image(boxes_per_image, image_shape)
            probs = F.softmax(logits_per_image, dim=1)

            picked_boxes = []
            picked_scores = []
            picked_labels = []

            for class_idx in range(1, self.num_classes):
                scores = probs[:, class_idx]
                keep = torch.nonzero(scores > self.score_thresh).squeeze(1)
                if keep.numel() == 0:
                    continue

                picked_boxes.append(boxes_per_image[keep])
                picked_scores.append(scores[keep])
                picked_labels.append(
                    torch.full((keep.numel(),), class_idx, dtype=torch.int64, device=boxes_per_image.device)
                )

            if not picked_boxes:
                detections.append(
                    {
                        "boxes": torch.zeros((0, 4), device=boxes_per_image.device),
                        "scores": torch.zeros((0,), device=boxes_per_image.device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=boxes_per_image.device),
                    }
                )
                continue

            merged_boxes = torch.cat(picked_boxes, dim=0)
            merged_scores = torch.cat(picked_scores, dim=0)
            merged_labels = torch.cat(picked_labels, dim=0)

            keep = box_ops.remove_small_boxes(merged_boxes, min_size=1.0)
            merged_boxes = merged_boxes[keep]
            merged_scores = merged_scores[keep]
            merged_labels = merged_labels[keep]

            keep = box_ops.batched_nms(merged_boxes, merged_scores, merged_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": merged_boxes[keep],
                    "scores": merged_scores[keep],
                    "labels": merged_labels[keep],
                }
            )

        return detections, {}


def build_cascade_frcnn(num_classes: int, min_size=600, max_size=1000):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )

    rpn = model.rpn
    if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
        rpn._pre_nms_top_n["training"] = 1000
        rpn._pre_nms_top_n["testing"] = 1000
    if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
        rpn._post_nms_top_n["training"] = 500
        rpn._post_nms_top_n["testing"] = 500

    feature_channels = getattr(model.backbone, "out_channels", 256)

    model.roi_heads = CascadeRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        num_classes=num_classes,
        feat_channels=feature_channels,
        num_stages=3,
        stage_bbox_stds=((0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067)),
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )
    return model


class DetectorWrapper:
    def __init__(self, model, weights_state_dict_path: str, device: str = "cuda"):
        use_cuda = torch.cuda.is_available() and device.startswith("cuda")
        self.device = torch.device(device if use_cuda else "cpu")

        checkpoint = torch.load(weights_state_dict_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        skip_prefixes = ("roi_heads.box_head.", "roi_heads.box_predictor.")
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith(skip_prefixes)}

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Loaded weights.")
        print("Missing keys:", len(missing_keys))
        if missing_keys:
            print("\n--- Missing keys ---")
            for key in missing_keys:
                print(key)

        print("\n--- Unexpected keys ---")
        for key in unexpected_keys:
            print(key)

        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_one(self, img_path: str, score_thr: float = 0.25) -> Prediction:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(img_path)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        output = self.model([image_tensor])[0]
        boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = output["scores"].detach().cpu().numpy().astype(np.float32)
        labels = output["labels"].detach().cpu().numpy().astype(np.int64)

        keep = scores >= score_thr
        return Prediction(boxes=boxes[keep], scores=scores[keep], labels=labels[keep])
