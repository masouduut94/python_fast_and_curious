from typing import List, Tuple
import json
import numpy as np
from numba import jit
from time import time


@jit(nogil=True, nopython=True)
def calculate_iou(x1, y1, x2, y2, x1_other, y1_other, x2_other, y2_other) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1_inter = max(x1, x1_other)
    y1_inter = max(y1, y1_other)
    x2_inter = min(x2, x2_other)
    y2_inter = min(y2, y2_other)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_other - x1_other) * (y2_other - y1_other)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


@jit(nogil=True, nopython=True)
def is_true_positive_or_false_positive(pred_box, gt_boxes, iou_threshold=0.5) -> bool:
    for gt_box in gt_boxes:
        if gt_box[2] == pred_box[2] and gt_box[1] == pred_box[1]:  # Compare category_id and image_id
            iou = calculate_iou(pred_box[3], pred_box[4], pred_box[5], pred_box[6], gt_box[3], gt_box[4], gt_box[5],
                                gt_box[6])
            if iou >= iou_threshold:
                return True
    return False


@jit(nogil=True, nopython=True)
def is_false_negative(gt_box, pred_boxes, iou_threshold=0.5) -> bool:
    for pred_box in pred_boxes:
        if pred_box[2] == gt_box[2] and pred_box[1] == gt_box[1]:  # Compare category_id and image_id
            iou = calculate_iou(
                gt_box[3], gt_box[4], gt_box[5], gt_box[6], pred_box[3], pred_box[4], pred_box[5], pred_box[6]
            )
            if iou >= iou_threshold:
                return False
    return True


def load_boxes_from_json(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    return np.array(
        [(
            ann['annotation_id'],
            ann['image_id'], ann['category_id'],
            ann['x1'],
            ann['y1'],
            ann['x1'] + ann['w'],
            ann['y1'] + ann['h'])
            for ann in data['annotations']],
        dtype=np.float32
    )


@jit(nogil=True, nopython=True)
def evaluate(gt_boxes: np.ndarray,
             pred_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp_pred_ids = np.empty(len(pred_boxes), dtype=np.int32)
    fp_pred_ids = np.empty(len(pred_boxes), dtype=np.int32)
    fn_gt_ids = np.empty(len(gt_boxes), dtype=np.int32)

    tp_pred_count = 0
    fp_pred_count = 0
    fn_gt_count = 0

    # Evaluate predictions
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        is_tp = is_true_positive_or_false_positive(pred_box, gt_boxes)
        if is_tp:
            tp_pred_ids[tp_pred_count] = pred_box[0]
            tp_pred_count += 1
        else:
            fp_pred_ids[fp_pred_count] = pred_box[0]
            fp_pred_count += 1

    # Evaluate ground truth for false negatives
    for i in range(len(gt_boxes)):
        gt_box = gt_boxes[i]
        if is_false_negative(gt_box, pred_boxes):
            fn_gt_ids[fn_gt_count] = gt_box[0]
            fn_gt_count += 1

    return tp_pred_ids[:tp_pred_count], fp_pred_ids[:fp_pred_count], fn_gt_ids[:fn_gt_count]


def measure_jit_evaluator_time(gt_json_path, pred_json_path, input_tp_ids, input_fp_ids, input_fn_ids):
    ground_truth_boxes = load_boxes_from_json(gt_json_path)
    predicted_boxes = load_boxes_from_json(pred_json_path)

    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluate(ground_truth_boxes, predicted_boxes)
    t2 = time()
    print(f"JIT | Execution Time: {t2 - t1: .4f} seconds")
    assert sorted(tp_ids) == sorted(input_tp_ids), "TP does not match!"
    assert sorted(fp_ids) == sorted(input_fp_ids), "FP does not match!"
    assert sorted(fn_ids) == sorted(input_fn_ids), "FN does not match!"


if __name__ == "__main__":
    import time

    start_time = time.time()
    ground_truth_boxes = load_boxes_from_json("../small_jsons/ground_truths.json")
    predicted_boxes = load_boxes_from_json("../small_jsons/predictions.json")
    tp_ids, fp_ids, fn_ids = evaluate(ground_truth_boxes, predicted_boxes)

    print("Execution Time: %s seconds" % (time.time() - start_time))
    print("True Positive Predictions:", len(tp_ids))
    print("False Positive Predictions:", len(fp_ids))
    print("False Negative Ground Truths:", len(fn_ids))
