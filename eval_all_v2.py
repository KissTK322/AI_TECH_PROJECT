import argparse
import time
from pathlib import Path

from evaluator_v2 import run_full_evaluation
from models_v2 import (
    DetectorWrapper,
    YOLOPredictor,
    build_cascade_frcnn,
    build_faster_rcnn,
)


def count_lines_in_class_file(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--img_dir", required=True)

    parser.add_argument("--yolo8_best", default=None)
    parser.add_argument("--yolo11_best", default=None)
    parser.add_argument("--frcnn_best", default=None)
    parser.add_argument("--cascade_best", default=None)

    parser.add_argument("--classes_txt", default=None)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--score_thr", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--limit_images",
        type=int,
        default=None,
        help="Use only part of the test set for quick checking",
    )
    return parser.parse_args()


def get_torchvision_num_classes(args) -> int | None:
    if args.classes_txt:
        return count_lines_in_class_file(args.classes_txt) + 1

    if args.num_classes is not None:
        return int(args.num_classes) + 1

    return None


def main():
    args = parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    output_root = Path("runs_eval") / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    tv_num_classes = get_torchvision_num_classes(args)
    reports = []

    if args.yolo8_best:
        predictor = YOLOPredictor(args.yolo8_best, device=args.device)
        reports.append(
            run_full_evaluation(
                model_name="YOLOv8s",
                coco_json=args.coco_json,
                img_dir=args.img_dir,
                predictor_fn=predictor.predict_one,
                out_dir=str(output_root / "yolov8s"),
                score_thr=args.score_thr,
                iou_thr_cm=args.iou_thr,
                limit_images=args.limit_images,
            )
        )

    if args.yolo11_best:
        predictor = YOLOPredictor(args.yolo11_best, device=args.device)
        reports.append(
            run_full_evaluation(
                model_name="YOLO11s",
                coco_json=args.coco_json,
                img_dir=args.img_dir,
                predictor_fn=predictor.predict_one,
                out_dir=str(output_root / "yolo11s"),
                score_thr=args.score_thr,
                iou_thr_cm=args.iou_thr,
                limit_images=args.limit_images,
            )
        )

    if args.frcnn_best:
        if tv_num_classes is None:
            raise SystemExit("For --frcnn_best, please provide --classes_txt or --num_classes.")

        model = build_faster_rcnn(num_classes=tv_num_classes, pretrained=False)
        predictor = DetectorWrapper(model, args.frcnn_best, device=args.device)
        reports.append(
            run_full_evaluation(
                model_name="FasterRCNN",
                coco_json=args.coco_json,
                img_dir=args.img_dir,
                predictor_fn=predictor.predict_one,
                out_dir=str(output_root / "faster_rcnn"),
                score_thr=args.score_thr,
                iou_thr_cm=args.iou_thr,
                limit_images=args.limit_images,
            )
        )

    if args.cascade_best:
        if tv_num_classes is None:
            raise SystemExit("For --cascade_best, please provide --classes_txt or --num_classes.")

        model = build_cascade_frcnn(num_classes=tv_num_classes)
        model.roi_heads.score_thresh = args.score_thr
        predictor = DetectorWrapper(model, args.cascade_best, device=args.device)
        reports.append(
            run_full_evaluation(
                model_name="CascadeFRCNN",
                coco_json=args.coco_json,
                img_dir=args.img_dir,
                predictor_fn=predictor.predict_one,
                out_dir=str(output_root / "cascade_frcnn"),
                score_thr=args.score_thr,
                iou_thr_cm=args.iou_thr,
                limit_images=args.limit_images,
            )
        )

    print("\n=== DONE ===")
    for report in reports:
        print(
            f"[{report['model']}] "
            f"mAP50-95={report['coco_map']['mAP_50_95']:.4f}  "
            f"mAP50={report['coco_map']['mAP_50']:.4f}  "
            f"mIoU(TP)={report['miou_tp_mean']:.4f}  "
            f"macroF1@iou={report['macro_f1@iou']:.4f}  "
            f"ms/img={report['inference_ms_per_image']:.1f}"
        )


if __name__ == "__main__":
    main()
