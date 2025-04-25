import os
import numpy as np
import sys
sys.path.append('./')
from vis_tools.engine import Detector
from tqdm import tqdm
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CLASSES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes in [x, y, w, h] format."""
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def evaluate_detector(detector, class_names):
    class_iou_scores = defaultdict(list)
    class_acc_05 = defaultdict(int)
    class_acc_075 = defaultdict(int)
    class_total = defaultdict(int)

    for idx in tqdm(range(detector.dataset.__len__()), desc="Evaluating"):
        outputs, image_path, caption, gt_box, pred_boxes, gt_cls = detector.infrence(idx)
        try:
            pred_box = list(pred_boxes[0,:])
            gt_box = list(gt_box[0,:])
            iou = compute_iou(pred_box, gt_box)
        except:
            iou = 0
        class_iou_scores[gt_cls].append(iou)

        if iou >= 0.5:
            class_acc_05[gt_cls] += 1
        if iou >= 0.75:
            class_acc_075[gt_cls] += 1

        class_total[gt_cls] += 1

    # Aggregate class-wise results
    classwise_results = {}
    for cls in class_names:
        total = class_total[cls]
        classwise_results[cls] = {
            "Acc@0.5": class_acc_05[cls] / total if total > 0 else 0.0,
            "Acc@0.75": class_acc_075[cls] / total if total > 0 else 0.0,
            "Mean IoU": float(np.mean(class_iou_scores[cls])) if total > 0 else 0.0,
            "Samples": total
        }

    # Compute overall metrics as average of class-wise results (weighted by sample count)
    total_samples = sum(class_total.values())
    weighted_acc_05 = sum(class_acc_05[cls] for cls in class_names) / total_samples if total_samples > 0 else 0.0
    weighted_acc_075 = sum(class_acc_075[cls] for cls in class_names) / total_samples if total_samples > 0 else 0.0
    all_ious = [iou for cls in class_names for iou in class_iou_scores[cls]]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    overall_results = {
        "Acc@0.5": weighted_acc_05,
        "Acc@0.75": weighted_acc_075,
        "Mean IoU": mean_iou,
        "Samples": total_samples
    }

    return overall_results, classwise_results


# Run evaluation
detector = Detector()
overall_results, classwise_results = evaluate_detector(detector, CLASSES)

print("Overall Results:", overall_results)
print("Class-wise Results:")
for cls, res in classwise_results.items():
    print(f"  {cls}: {res}")


# Overall Results: {'Acc@0.5': 0.938526258651106, 'Acc@0.75': 0.9109784231238974, 'Mean IoU': 0.8863711968518785, 'Samples': 7369}
# Class-wise Results:
#             pedestrian: {'Acc@0.5': 0.9309989701338826, 'Acc@0.75': 0.893923789907312, 'Mean IoU': 0.8622691644629968, 'Samples': 971}
#             rider: {'Acc@0.5': 0.9563812600969306, 'Acc@0.75': 0.9499192245557351, 'Mean IoU': 0.8988507858482047, 'Samples': 619}
#             car: {'Acc@0.5': 0.9380836378400325, 'Acc@0.75': 0.9092570036540804, 'Mean IoU': 0.8929239570337506, 'Samples': 4926}
#             bus: {'Acc@0.5': 0.8988095238095238, 'Acc@0.75': 0.8928571428571429, 'Mean IoU': 0.8475329542443866, 'Samples': 168}
#             truck: {'Acc@0.5': 0.8854961832061069, 'Acc@0.75': 0.8358778625954199, 'Mean IoU': 0.8104459343776675, 'Samples': 262}
#             bicycle: {'Acc@0.5': 0.9817708333333334, 'Acc@0.75': 0.9635416666666666, 'Mean IoU': 0.9071115298817555, 'Samples': 384}
#             motorcycle: {'Acc@0.5': 1.0, 'Acc@0.75': 1.0, 'Mean IoU': 0.9338651895523071, 'Samples': 39}


# Overall Results: {'Acc@0.5': 0.8269778803094042, 'Acc@0.75': 0.7618401411317682, 'Mean IoU': 0.7896174564240707, 'Samples': 7369}
# Class-wise Results:
#   pedestrian: {'Acc@0.5': 0.8280123583934088, 'Acc@0.75': 0.7342945417095778, 'Mean IoU': 0.7752674998318437, 'Samples': 971}
#   rider: {'Acc@0.5': 0.9418416801292407, 'Acc@0.75': 0.8998384491114702, 'Mean IoU': 0.8801466668135997, 'Samples': 619}
#   car: {'Acc@0.5': 0.8004466098254162, 'Acc@0.75': 0.7332521315468941, 'Mean IoU': 0.7724711785152599, 'Samples': 4926}
#   bus: {'Acc@0.5': 0.9880952380952381, 'Acc@0.75': 0.9821428571428571, 'Mean IoU': 0.9367626000727926, 'Samples': 168}
#   truck: {'Acc@0.5': 0.7709923664122137, 'Acc@0.75': 0.7137404580152672, 'Mean IoU': 0.735210393164449, 'Samples': 262}
#   bicycle: {'Acc@0.5': 0.9348958333333334, 'Acc@0.75': 0.8958333333333334, 'Mean IoU': 0.8633801335624108, 'Samples': 384}
#   motorcycle: {'Acc@0.5': 0.9487179487179487, 'Acc@0.75': 0.9230769230769231, 'Mean IoU': 0.8811095081842862, 'Samples': 39}


# Overall Results: {'Acc@0.5': 0.7873450750163079, 'Acc@0.75': 0.7395955642530985, 'Mean IoU': 0.734922285008146, 'Samples': 7665}
# Class-wise Results:
#   pedestrian: {'Acc@0.5': 0.7611788617886179, 'Acc@0.75': 0.698170731707317, 'Mean IoU': 0.6972565912651996, 'Samples': 984}
#   rider: {'Acc@0.5': 0.7971698113207547, 'Acc@0.75': 0.7908805031446541, 'Mean IoU': 0.7326377166036822, 'Samples': 636}
#   car: {'Acc@0.5': 0.7847753580537571, 'Acc@0.75': 0.7427898763978811, 'Mean IoU': 0.7415527090207136, 'Samples': 5097}
#   bus: {'Acc@0.5': 0.8035714285714286, 'Acc@0.75': 0.7976190476190477, 'Mean IoU': 0.7590656546609742, 'Samples': 168}
#   truck: {'Acc@0.5': 0.7592592592592593, 'Acc@0.75': 0.7148148148148148, 'Mean IoU': 0.7026848534811978, 'Samples': 270}
#   bicycle: {'Acc@0.5': 0.8598726114649682, 'Acc@0.75': 0.7048832271762208, 'Mean IoU': 0.7495869510903986, 'Samples': 471}
#   motorcycle: {'Acc@0.5': 0.8717948717948718, 'Acc@0.75': 0.8717948717948718, 'Mean IoU': 0.7980433014722971, 'Samples': 39}