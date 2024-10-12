import json
import numpy as np
cimport cython


@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative index wraparound for performance
cdef class Evaluator:
    cdef float[:, :] ground_truth_boxes
    cdef float[:, :] predicted_boxes
    cdef int[:] gt_annotation_ids
    cdef int[:] pred_annotation_ids

    def __init__(self, str ground_truth_json, str predictions_json):
        cdef int num_gt, num_pred

        with open(ground_truth_json, 'r') as f:
            ground_truth_data = json.load(f)
        with open(predictions_json, 'r') as f:
            predictions_data = json.load(f)

        num_gt = len(ground_truth_data['annotations'])
        num_pred = len(predictions_data['annotations'])

        self.ground_truth_boxes = np.zeros((num_gt, 8), dtype=np.float32)
        self.predicted_boxes = np.zeros((num_pred, 8), dtype=np.float32)
        self.gt_annotation_ids = np.zeros(num_gt, dtype=np.int32)
        self.pred_annotation_ids = np.zeros(num_pred, dtype=np.int32)

        self._initialize_boxes(self.ground_truth_boxes, self.gt_annotation_ids, ground_truth_data['annotations'])
        self._initialize_boxes(self.predicted_boxes, self.pred_annotation_ids, predictions_data['annotations'])

    cdef void _initialize_boxes(self, float[:, :] boxes, int[:] annotation_ids, list data):
        cdef int i
        for i in range(boxes.shape[0]):
            annotation_ids[i] = data[i]['annotation_id']
            boxes[i, 0] = data[i]['image_id']
            boxes[i, 1] = data[i]['category_id']
            boxes[i, 2] = data[i]['bbox'][0]
            boxes[i, 3] = data[i]['bbox'][1]
            boxes[i, 4] = data[i]['bbox'][0] + data[i]['bbox'][2]
            boxes[i, 5] = data[i]['bbox'][1] + data[i]['bbox'][3]
            boxes[i, 6] = data[i]['bbox'][2]
            boxes[i, 7] = data[i]['bbox'][3]

    cpdef tuple evaluate(self):
        cdef list tp_pred_ids = []
        cdef list fp_pred_ids = []
        cdef list fn_gt_ids = []

        cdef int i, j
        cdef float iou
        cdef bint is_tp

        for i in range(self.predicted_boxes.shape[0]):
            is_tp = self._is_true_positive_or_false_positive(self.predicted_boxes[i, :], self.ground_truth_boxes)
            if is_tp:
                tp_pred_ids.append(self.pred_annotation_ids[i])
            else:
                fp_pred_ids.append(self.pred_annotation_ids[i])

        for i in range(self.ground_truth_boxes.shape[0]):
            if self._is_false_negative(self.ground_truth_boxes[i, :], self.predicted_boxes):
                fn_gt_ids.append(self.gt_annotation_ids[i])

        return tp_pred_ids, fp_pred_ids, fn_gt_ids

    cdef tuple _is_true_positive_or_false_positive(self, float[:] pred_box, float[:, :] ground_truth_boxes, float iou_threshold=0.5):
        cdef int i
        cdef float iou

        for i in range(ground_truth_boxes.shape[0]):
            if ground_truth_boxes[i, 1] == pred_box[1] and ground_truth_boxes[i, 0] == pred_box[0]:
                iou = self._calculate_iou(pred_box, ground_truth_boxes[i, :])
                if iou >= iou_threshold:
                    return True
        return False

    cdef bint _is_false_negative(self, float[:] gt_box, float[:, :] predicted_boxes, float iou_threshold=0.5):
        cdef int i
        cdef float iou

        for i in range(predicted_boxes.shape[0]):
            if predicted_boxes[i, 1] == gt_box[1] and predicted_boxes[i, 0] == gt_box[0]:
                iou = self._calculate_iou(gt_box, predicted_boxes[i, :])
                if iou >= iou_threshold:
                    return False
        return True

    cdef float _calculate_iou(self, float[:] box1, float[:] box2):
        cdef float x1_inter = max(box1[2], box2[2])
        cdef float y1_inter = max(box1[3], box2[3])
        cdef float x2_inter = min(box1[4], box2[4])
        cdef float y2_inter = min(box1[5], box2[5])

        cdef float inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        cdef float box1_area = box1[6] * box1[7]
        cdef float box2_area = box2[6] * box2[7]

        return inter_area / (box1_area + box2_area - inter_area)

