import taichi as ti
from pathlib import Path
from time import time
import json

ti.init(arch=ti.cpu)


@ti.data_oriented
def initialize_boxes(annotations, boxes):
    for i, ann in enumerate(annotations):
        boxes[i] = [ann['x1'], ann['y1'], ann['x1'] + ann['w'], ann['y1'] + ann['h'], ann['category_id'],
                    ann['annotation_id']]


@ti.func
def calculate_iou(box1, box2) -> ti.f32:
    x1_inter = ti.max(box1[0], box2[0])
    y1_inter = ti.max(box1[1], box2[1])
    x2_inter = ti.min(box1[2], box2[2])
    y2_inter = ti.min(box1[3], box2[3])

    inter_area = ti.max(0, x2_inter - x1_inter) * ti.max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


@ti.kernel
def evaluate(num_gt_boxes: int, num_pred_boxes: int, gt_boxes: ti.template(), pred_boxes: ti.template(),
             tp_ids: ti.template(), fp_ids: ti.template(), fn_ids: ti.template()):
    for i in range(num_pred_boxes):
        pred_box = pred_boxes[i]
        is_tp = False

        for j in range(num_gt_boxes):
            gt_box = gt_boxes[j]
            if pred_box[4] == gt_box[4]:  # Same category
                iou = calculate_iou(pred_box, gt_box)
                if iou >= 0.5:
                    is_tp = True
                    break

        if is_tp:
            tp_ids[i] = pred_box[5]
        else:
            fp_ids[i] = pred_box[5]

    for j in range(num_gt_boxes):
        gt_box = gt_boxes[j]
        is_fn = True

        for i in range(num_pred_boxes):
            pred_box = pred_boxes[i]
            if pred_box[4] == gt_box[4]:  # Same category
                iou = calculate_iou(pred_box, gt_box)
                if iou >= 0.5:
                    is_fn = False
                    break

        if is_fn:
            fn_ids[j] = gt_box[5]


def taichi_evaluate(ground_truth_json: str, predictions_json: str):
    assert Path(ground_truth_json).is_file(), 'Ground truth JSON file not found.'
    assert Path(predictions_json).is_file(), 'Prediction JSON file not found.'

    with open(ground_truth_json) as f:
        ground_truth_data = json.load(f)
    with open(predictions_json) as f:
        predictions_data = json.load(f)

    num_gt_boxes = len(ground_truth_data['annotations'])
    num_pred_boxes = len(predictions_data['annotations'])

    t1 = time()
    gt_boxes = ti.Vector.field(6, dtype=ti.i32, shape=num_gt_boxes)
    pred_boxes = ti.Vector.field(6, dtype=ti.i32, shape=num_pred_boxes)
    tp_ids = ti.field(dtype=ti.i32, shape=num_pred_boxes)
    fp_ids = ti.field(dtype=ti.i32, shape=num_pred_boxes)
    fn_ids = ti.field(dtype=ti.i32, shape=num_gt_boxes)
    initialize_boxes(ground_truth_data['annotations'], gt_boxes)
    initialize_boxes(predictions_data['annotations'], pred_boxes)
    evaluate(num_gt_boxes, num_pred_boxes, gt_boxes, pred_boxes, tp_ids, fp_ids, fn_ids)
    t2 = time()
    return t2 - t1, tp_ids.to_numpy(), fp_ids.to_numpy(), fn_ids.to_numpy()


def measure_taichi_evaluator_time(gt_json_path, pred_json_path):
    elapsed_time, tp_ids, fp_ids, fn_ids = taichi_evaluate(gt_json_path, pred_json_path)
    tp_ids = [tp for tp in tp_ids if tp != 0]
    fp_ids = [fp for fp in fp_ids if fp != 0]
    fn_ids = [fn for fn in fn_ids if fn != 0]
    return elapsed_time, sorted(tp_ids), sorted(fp_ids), sorted(fn_ids)
