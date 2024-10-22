import json
from pathlib import Path
from typing import List, Self


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
            gt_data = json.load(f)
        with open(predictions_json) as f:
            pred_data = json.load(f)

        self.ground_truth_boxes = []
        self.predicted_boxes = []

        for gt in gt_data['annotations']:
            bb = BoundingBox(
                annotation_id=gt['annotation_id'],
                image_id=gt['image_id'],
                category_id=gt['category_id'],
                x1=gt['bbox'][0],
                y1=gt['bbox'][1],
                w=gt['bbox'][2],
                h=gt['bbox'][3]
            )
            self.ground_truth_boxes.append(bb)

        for pred in pred_data['annotations']:
            bb = BoundingBox(
                annotation_id=pred['annotation_id'],
                image_id=pred['image_id'],
                category_id=pred['category_id'],
                x1=pred['bbox'][0],
                y1=pred['bbox'][1],
                w=pred['bbox'][2],
                h=pred['bbox'][3]
            )
            self.predicted_boxes.append(bb)

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
