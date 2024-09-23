from time import time

import matplotlib.pyplot as plt
from matplotlib import patches

from pathlib import Path
from typing import List, Self
import json


class BoundingBox:
    """

    Attributes:
        annotation_id (int):
        image_id (int):
        category_id (int):
        x1 (int):
        y1 (int):
        x2 (int):
        y2 (int):
        w (int):
        h (int):

    Methods:



    """

    def __init__(self, annotation_id: int, image_id: int, category_id: int, x1: int, y1: int, w: int, h: int):
        """

        Args:
            annotation_id:
            image_id:
            category_id:
            x1:
            y1:
            w:
            h:

        """
        self.annotation_id = annotation_id
        self.image_id = image_id
        self.category_id = category_id
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.x2 = x1 + w
        self.y2 = y1 + h

    def is_true_positive_or_false_positive(self, ground_truth_boxes: List[Self], iou_threshold=0.5):
        """
        Determines if the prediction bounding-box is a True Positive or False Positive.
        Returns:
            Tuple[bool, int | None]: whether detection is true positive or false positive, and matched_annotation_id
        """
        for gt_box in ground_truth_boxes:
            if gt_box.category_id == self.category_id and gt_box.image_id == self.image_id:
                iou = self.calculate_iou(gt_box)
                if iou >= iou_threshold:
                    return True
        return False

    def is_false_negative(self, predicted_boxes: List[Self], iou_threshold: float = 0.5):
        """
        Determines if a ground truth bounding-box is a False Negative or not.
        Returns:
            bool: whether the bounding-box is a false negative or not.
        """
        for pred_box in predicted_boxes:
            if pred_box.category_id == self.category_id and pred_box.image_id == self.image_id:
                iou = self.calculate_iou(pred_box)
                if iou >= iou_threshold:
                    return False
        return True

    def calculate_iou(self, other_box: Self):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        """
        x1_inter = max(self.x1, other_box.x1)
        y1_inter = max(self.y1, other_box.y1)
        x2_inter = min(self.x2, other_box.x2)
        y2_inter = min(self.y2, other_box.y2)

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = self.w * self.h
        box2_area = other_box.w * other_box.h

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou


class Evaluator:
    def __init__(self, ground_truth_json: str, predictions_json: str):
        assert Path(ground_truth_json).is_file(), 'ground truth json file not found.'
        assert Path(predictions_json).is_file(), 'prediction json file not found.'

        with open(ground_truth_json) as f:
            ground_truth_data = json.load(f)
        with open(predictions_json) as f:
            predictions_data = json.load(f)

        self.ground_truth_boxes = [BoundingBox(**ann) for ann in ground_truth_data['annotations']]
        self.predicted_boxes = [BoundingBox(**ann) for ann in predictions_data['annotations']]

    def evaluate(self):
        """
        Loops through both prediction bboxes and ground truth bboxes, to find
        False Positives, False Negatives, and True Positives.

        Returns:
            Tuple[List[int], List[int], List[int]]
        """
        tp_ids: List[int] = []
        fp_ids: List[int] = []
        fn_ids: List[int] = []

        for pred_box in self.predicted_boxes:
            is_tp = pred_box.is_true_positive_or_false_positive(self.ground_truth_boxes)
            if is_tp:
                tp_ids.append(pred_box.annotation_id)
            else:
                fp_ids.append(pred_box.annotation_id)

        for gt_box in self.ground_truth_boxes:
            if gt_box.is_false_negative(self.predicted_boxes):
                fn_ids.append(gt_box.annotation_id)

        return tp_ids, fp_ids, fn_ids


def plot_results(ground_truth_boxes, predicted_boxes, tp_ids: List[int], fp_ids: List[int],
                 fn_ids: List[int], image_width: int, image_height: int):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)

    for bbox in ground_truth_boxes:
        if bbox.annotation_id in fn_ids:
            color = 'yellow'
            text = 'FN'
            rectangle = patches.Rectangle(
                (bbox.x1, bbox.y1),
                bbox.w,
                bbox.h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rectangle)

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax.annotate(
                text,
                (cx, cy),
                color=color,
                weight='bold',
                fontsize=10,
                ha='center',
                va='center'
            )

    for bbox in predicted_boxes:
        if bbox.annotation_id in fp_ids:
            color = 'red'
            text = 'FP'
            rectangle = patches.Rectangle(
                (bbox.x1, bbox.y1),
                bbox.w,
                bbox.h,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='dashed'
            )
            ax.add_patch(rectangle)

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax.annotate(
                text,
                (cx, cy),
                color=color,
                weight='bold',
                fontsize=10,
                ha='center',
                va='center'
            )
        elif bbox.annotation_id in tp_ids:
            color = 'green'
            text = 'TP'
            rectangle = patches.Rectangle(
                (bbox.x1, bbox.y1),
                bbox.w,
                bbox.h,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='dashed'
            )
            ax.add_patch(rectangle)

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax.annotate(
                text,
                (cx, cy),
                color=color,
                weight='bold',
                fontsize=10,
                ha='center',
                va='center'
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y-axis to match typical image coordinates
    plt.show()

def measure_py_evaluator_time(gt_json_path, pred_json_path):
    evaluator = Evaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator.evaluate()
    t2 = time()
    print(f"Execution time: {t2 - t1:.4f} seconds")
    return tp_ids, fp_ids, fn_ids

