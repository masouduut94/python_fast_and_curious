from evaluator import Evaluator  # Import the C++ module
from time import time


def measure_parallel_cython_evaluator(gt_json_path, pred_json_path, input_tp_ids, input_fp_ids, input_fn_ids):
    evaluator_instance = Evaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator_instance.evaluate()
    t2 = time()

    print(f"Parallel Cython | Execution Time: {t2 - t1: .4f} seconds.")
    assert sorted(input_tp_ids) == sorted(tp_ids), "tp_ids don't match!"
    assert sorted(input_fp_ids) == sorted(fp_ids), "fp_ids don't match!"
    assert sorted(input_fn_ids) == sorted(fn_ids), "fn_ids don't match!"


if __name__ == '__main__':
    gt_json_path = '/jsons_small/ground_truths.json'
    pred_json_path = '/jsons_small/predictions.json'
    evaluator_instance = Evaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator_instance.evaluate()
    t2 = time()