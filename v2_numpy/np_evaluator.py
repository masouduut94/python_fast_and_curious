"""
Replace BoundingBox class with numpy array to speed up.

"""
import json
from typing import List, Tuple

import numpy as np
from pathlib import Path
from time import time

from numpy.typing import NDArray


class Evaluator:
    def __init__(self, ground_truth_json: str, predictions_json: str):
        assert Path(ground_truth_json).is_file(), 'Ground truth JSON file not found.'
        assert Path(predictions_json).is_file(), 'Prediction JSON file not found.'

        with open(ground_truth_json) as f:
            ground_truth_data = json.load(f)
        with open(predictions_json) as f:
            predictions_data = json.load(f)

        # Convert annotations to numpy arrays
        self.ground_truth_boxes = np.array([
            [ann['annotation_id'], ann['image_id'], ann['category_id'], ann['x1'], ann['y1'], ann['x1'] + ann['w'],
             ann['y1'] + ann['h']]
            for ann in ground_truth_data['annotations']
        ])

        self.predicted_boxes = np.array([
            [ann['annotation_id'], ann['image_id'], ann['category_id'], ann['x1'], ann['y1'], ann['x1'] + ann['w'],
             ann['y1'] + ann['h']]
            for ann in predictions_data['annotations']
        ])

    @staticmethod
    def calculate_iou(box1: NDArray, box2: NDArray) -> float:
        # Unpack the boxes
        x1_1, y1_1, x2_1, y2_1 = box1[3], box1[4], box1[5], box1[6]
        x1_2, y1_2, x2_2, y2_2 = box2[3], box2[4], box2[5], box2[6]

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def evaluate(self, iou_threshold: float = 0.5) -> Tuple[List[int], List[int], List[int]]:
        tp_ids = []
        fp_ids = []
        fn_ids = []

        for pred_box in self.predicted_boxes:
            # Check for matching ground truth box
            gt_matches = self.ground_truth_boxes[
                (self.ground_truth_boxes[:, 1] == pred_box[1]) &  # Match image_id
                (self.ground_truth_boxes[:, 2] == pred_box[2])  # Match category_id
                ]
            matched = False
            for gt_box in gt_matches:
                iou = self.calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp_ids.append(pred_box[0])
                    matched = True
                    break
            if not matched:
                fp_ids.append(pred_box[0])

        for gt_box in self.ground_truth_boxes:
            pred_matches = self.predicted_boxes[
                (self.predicted_boxes[:, 1] == gt_box[1]) &  # Match image_id
                (self.predicted_boxes[:, 2] == gt_box[2])  # Match category_id
                ]
            matched = False
            for pred_box in pred_matches:
                iou = self.calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched = True
                    break
            if not matched:
                fn_ids.append(gt_box[0])

        return tp_ids, fp_ids, fn_ids


def measure_np_evaluator_time(gt_json_path: str, pred_json_path: str):
    evaluator = Evaluator(ground_truth_json=gt_json_path, predictions_json=pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator.evaluate()
    t2 = time()

    return t2 - t1, sorted(tp_ids), sorted(fp_ids), sorted(fn_ids)

