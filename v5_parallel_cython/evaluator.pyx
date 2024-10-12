import json
cimport cython
from cython.parallel import prange
from cython.view cimport array
import numpy as np
cimport numpy as np

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class Evaluator:
    cdef float[:, :] ground_truth_boxes
    cdef float[:, :] predicted_boxes
    cdef float[:] gt_annotation_ids
    cdef float[:] pred_annotation_ids

    __slots__ = ["ground_truth_boxes", "predicted_boxes", "gt_annotation_ids", "pred_annotation_ids"]

    def __init__(self, str ground_truth_json, str predictions_json):
        cdef int num_gt, num_pred

        with open(ground_truth_json, 'r') as f:
            ground_truth_data = json.load(f)
        with open(predictions_json, 'r') as f:
            predictions_data = json.load(f)

        num_gt = len(ground_truth_data['annotations'])
        num_pred = len(predictions_data['annotations'])

        # Initialize memoryviews
        self.ground_truth_boxes = self._create_array(num_gt, 8)
        self.predicted_boxes = self._create_array(num_pred, 8)
        self.gt_annotation_ids = self._create_float_array(num_gt)
        self.pred_annotation_ids = self._create_float_array(num_pred)

        # Populate memoryviews
        self._initialize_boxes(self.ground_truth_boxes, self.gt_annotation_ids, ground_truth_data['annotations'])
        self._initialize_boxes(self.predicted_boxes, self.pred_annotation_ids, predictions_data['annotations'])

    cdef float[:, :] _create_array(self, int rows, int cols):
        cdef float[:, :] arr = array(shape=(rows, cols), itemsize=sizeof(float), format='f')
        return arr

    cdef float[:] _create_float_array(self, int length):
        cdef float[:] arr = array(shape=(length,), itemsize=sizeof(float), format='f')
        return arr

    cdef void _initialize_boxes(self, float[:, :] boxes, float[:] annotation_ids, list data):
        cdef int i
        for i in range(len(data)):
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
        cdef int i, n_pred, n_gt
        cdef int tp_pred_index = 0
        cdef int fp_pred_index = 0
        cdef int fn_gt_index = 0

        n_pred = self.predicted_boxes.shape[0]
        n_gt = self.ground_truth_boxes.shape[0]

        # Initialize results with maximum length
        cdef float[:] tp_pred_ids = self._create_float_array(n_pred)
        cdef float[:] fp_pred_ids = self._create_float_array(n_pred)
        cdef float[:] fn_gt_ids = self._create_float_array(n_gt)
        cdef float output

        # Allocate thread-local storage for each thread
        cdef int[:] tp_pred_thread_ids = np.zeros(n_pred, dtype=np.int32)
        cdef int[:] fp_pred_thread_ids = np.zeros(n_pred, dtype=np.int32)
        cdef int[:] fn_gt_thread_ids = np.zeros(n_gt, dtype=np.int32)

        # Parallelize the computation using prange
        with nogil, cython.parallel.parallel():
            for i in prange(n_pred, schedule='dynamic'):
                output = self._is_true_positive_or_false_positive(self.predicted_boxes[i], self.ground_truth_boxes)
                if output:
                    # Each thread works on its own slice of the array
                    tp_pred_thread_ids[i] = 1  # Mark as true positive
                else:
                    fp_pred_thread_ids[i] = 1  # Mark as false positive

            # Process ground truth boxes
            for i in prange(n_gt, schedule='dynamic'):
                if self._is_false_negative(self.ground_truth_boxes[i], self.predicted_boxes):
                    fn_gt_thread_ids[i] = 1  # Mark as false negative

        # After the parallel block, process the results
        tp_pred_list = [self.pred_annotation_ids[i] for i in range(n_pred) if tp_pred_thread_ids[i]]
        fp_pred_list = [self.pred_annotation_ids[i] for i in range(n_pred) if fp_pred_thread_ids[i]]
        fn_gt_list = [self.gt_annotation_ids[i] for i in range(n_gt) if fn_gt_thread_ids[i]]

        return tp_pred_list, fp_pred_list, fn_gt_list

    cdef bint _is_true_positive_or_false_positive(self, float[:] pred_box, float[:, :] ground_truth_boxes, float iou_threshold=0.5) nogil:
        cdef int i
        cdef float iou

        for i in range(ground_truth_boxes.shape[0]):
            if ground_truth_boxes[i, 1] == pred_box[1] and ground_truth_boxes[i, 0] == pred_box[0]:
                iou = self._calculate_iou(pred_box, ground_truth_boxes[i])
                if iou >= iou_threshold:
                    return True
        return False

    cdef bint _is_false_negative(self, float[:] gt_box, float[:, :] predicted_boxes, float iou_threshold=0.5) nogil:
        cdef int i
        cdef float iou

        for i in range(predicted_boxes.shape[0]):
            if predicted_boxes[i, 1] == gt_box[1] and predicted_boxes[i, 0] == gt_box[0]:
                iou = self._calculate_iou(gt_box, predicted_boxes[i])
                if iou >= iou_threshold:
                    return False
        return True

    cdef float _calculate_iou(self, float[:] box1, float[:] box2) nogil:
        cdef float x1_inter = max(box1[2], box2[2])
        cdef float y1_inter = max(box1[3], box2[3])
        cdef float x2_inter = min(box1[4], box2[4])
        cdef float y2_inter = min(box1[5], box2[5])

        cdef float inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        cdef float box1_area = box1[6] * box1[7]
        cdef float box2_area = box2[6] * box2[7]

        return inter_area / (box1_area + box2_area - inter_area)
