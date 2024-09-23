from openmp_evaluator import OpenmpEvaluator
from time import time

# Create an evaluator object
gt_json = '/home/masoud/Desktop/projects/iou_speed_check/large_jsons/ground_truths.json'
pred_json = '/home/masoud/Desktop/projects/iou_speed_check/large_jsons/predictions.json'
evaluator = OpenmpEvaluator(gt_json, pred_json)

t1 = time()
tp_pred_ids, fp_pred_ids, fn_gt_ids = evaluator.evaluate()
t2 = time()

print(f"Process time: {t2-t1: .5f}")

# print("True Positive IDs:", sorted(tp_pred_ids))
# print("False Positive IDs:", sorted(fp_pred_ids))
# print("False Negative IDs:", sorted(fn_gt_ids))