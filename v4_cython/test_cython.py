from typing import List

from .evaluator import Evaluator
from time import time


def measure_cython_evaluator_time(gt_json_path: str, pred_json_path: str, input_tp_ids: List[int],
                                  input_fp_ids: List[int], input_fn_ids: List[int]):
    evaluator_instance = Evaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator_instance.evaluate()
    t2 = time()

    print(f"Cython | Execution Time: {t2 - t1: .4f} seconds")

    assert sorted(input_tp_ids) == sorted(tp_ids), "tp_ids don't match!"
    assert sorted(input_fp_ids) == sorted(fp_ids), "fp_ids don't match!"
    assert sorted(input_fn_ids) == sorted(fn_ids), "fn_ids don't match!"
