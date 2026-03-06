import argparse, time
from pathlib import Path

from evaluator import run_full_evaluation
from models import YOLOUltralyticsWrapper, build_faster_rcnn, build_cascade_frcnn, TorchVisionDetectorWrapper

def count_classes_from_txt(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--img_dir", required=True)

    ap.add_argument("--yolo8_best", default=None)
    ap.add_argument("--yolo11_best", default=None)
    ap.add_argument("--frcnn_best", default=None)
    ap.add_argument("--cascade_best", default=None)

    ap.add_argument("--classes_txt", default=None)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--score_thr", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--limit_images", type=int, default=None, help="Limit number of test images for quick debug")
    args = ap.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path("runs_eval") / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    tv_num_classes = None
    if args.classes_txt:
        tv_num_classes = count_classes_from_txt(args.classes_txt) + 1
    elif args.num_classes is not None:
        tv_num_classes = int(args.num_classes) + 1

    summaries = []

    if args.yolo8_best:
        w = YOLOUltralyticsWrapper(args.yolo8_best, device=args.device)
        summaries.append(run_full_evaluation("YOLOv8s", args.coco_json, args.img_dir, w.predict_one, str(out_root/"yolov8s"),
                                             score_thr=args.score_thr, iou_thr_cm=args.iou_thr))

    if args.yolo11_best:
        w = YOLOUltralyticsWrapper(args.yolo11_best, device=args.device)
        summaries.append(run_full_evaluation("YOLO11s", args.coco_json, args.img_dir, w.predict_one, str(out_root/"yolo11s"),
                                             score_thr=args.score_thr, iou_thr_cm=args.iou_thr))

    if args.frcnn_best:
        if tv_num_classes is None:
            raise SystemExit("For --frcnn_best you must provide --classes_txt or --num_classes.")
        model = build_faster_rcnn(num_classes=tv_num_classes, pretrained=False)
        w = TorchVisionDetectorWrapper(model, args.frcnn_best, device=args.device)
        summaries.append(run_full_evaluation("FasterRCNN", args.coco_json, args.img_dir, w.predict_one, str(out_root/"faster_rcnn"),
                                             score_thr=args.score_thr, iou_thr_cm=args.iou_thr))

    if args.cascade_best:
        if tv_num_classes is None:
            raise SystemExit("For --cascade_best you must provide --classes_txt or --num_classes.")
        model = build_cascade_frcnn(num_classes=tv_num_classes)
        model.roi_heads.score_thresh = args.score_thr   # ✅ ทำให้ --score_thr มีผลจริง
        w = TorchVisionDetectorWrapper(model, args.cascade_best, device=args.device)
        summaries.append(run_full_evaluation(
            "CascadeFRCNN", args.coco_json, args.img_dir, w.predict_one, str(out_root/"cascade_frcnn"),
            score_thr=args.score_thr, iou_thr_cm=args.iou_thr,
            limit_images=args.limit_images
        ))

    print("\n=== DONE ===")
    for s in summaries:
        print(f"[{s['model']}] mAP50-95={s['coco_map']['mAP_50_95']:.4f}  mAP50={s['coco_map']['mAP_50']:.4f}  "
              f"mIoU(TP)={s['miou_tp_mean']:.4f}  macroF1@iou={s['macro_f1@iou']:.4f}  "
              f"ms/img={s['inference_ms_per_image']:.1f}")

if __name__ == "__main__":
    main()
