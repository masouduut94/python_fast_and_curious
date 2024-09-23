from .shared_mutex_parallel_evaluator import SharedMutexParallelCppEvaluator  # Import the compiled module
from time import time


def measure_shared_mutex_parallel_cpp_time(gt_json_path, pred_json_path, input_tp_ids, input_fp_ids, input_fn_ids):
    evaluator_instance = SharedMutexParallelCppEvaluator(gt_json_path, pred_json_path)
    t1 = time()
    tp_ids, fp_ids, fn_ids = evaluator_instance.evaluate()
    t2 = time()
    print(f"Parallel C++ (Shared Mutex) | Execution Time: {t2 - t1: .4f}")

    assert sorted(input_tp_ids) == sorted(tp_ids), "tp_ids don't match!"
    assert sorted(input_fp_ids) == sorted(fp_ids), "fp_ids don't match!"
    assert sorted(input_fn_ids) == sorted(fn_ids), "fn_ids don't match!"

