from .parallel_cpp_evaluator import ParallelCppEvaluator  # Import the compiled module from so file
from time import time


def measure_parallel_cpp_evaluator_time(gt_json_path, pred_json_path, input_tp_ids, input_fp_ids, input_fn_ids):
    evaluator_instance = ParallelCppEvaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator_instance.evaluate()
    t2 = time()
    print(f"Parallel C++ | Process Time: {t2 - t1: .4f} seconds")

    assert sorted(input_tp_ids) == sorted(tp_ids), "tp_ids don't match!"
    assert sorted(input_fp_ids) == sorted(fp_ids), "fp_ids don't match!"
    assert sorted(input_fn_ids) == sorted(fn_ids), "fn_ids don't match!"

