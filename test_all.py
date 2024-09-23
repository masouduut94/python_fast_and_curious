"""
On large jsons:
    Execution time: 123.1167 seconds
    Execution time: 474.6064 seconds
    JIT | Execution Time:  3.8558 seconds
    Tai-Chi | Execution Time:  2.2846 seconds
    CythonEvaluator |  7.60301 process time
    CyParallelEvaluator |  18.69666 process time
    C++ Evaluator |  0.63920 process time
    C++ Parallel Evaluator |  0.05071 process time
    C++ Parallel OpenMP |  0.05313 process time

"""


import time
from v1_python.py_evaluator import measure_py_evaluator_time
from v2_numpy.np_evaluator import measure_np_evaluator_time
from v3_jit_functional.jit_evaluator import measure_jit_evaluator_time
from v4_tai_chi.taichi_evaluator import measure_taichi_evaluator_time
from v5_cython.evaluator import Evaluator as CyEvaluator
from v6_parallel_cython.evaluator import Evaluator as CyParallelEvaluator
from v7_cpp.cpp_evaluator import CppEvaluator
from v8_cpp_parallel.parallel_cpp_evaluator import ParallelCppEvaluator
from v9_cpp_parallel_shared_mutex.shared_mutex_parallel_evaluator import SharedMutexParallelCppEvaluator

def check_evaluator_time(module, name, gt_json_path, pred_json_path, input_tp_ids, input_fp_ids, input_fn_ids):
    evaluator = module(gt_json_path, pred_json_path)
    t1 = time.time()
    tp_ids, fp_ids, fn_ids = evaluator.evaluate()
    t2 = time.time()
    print(f"{name} | {t2 - t1: .5f} process time")
    assert sorted(input_tp_ids) == sorted(tp_ids), "TP ID issue!"
    assert sorted(input_fp_ids) == sorted(fp_ids), "FP ID issue!"
    assert sorted(input_fn_ids) == sorted(fn_ids), "FN ID issue!"



if __name__ == "__main__":
    gt_json_path = 'large_jsons/ground_truths.json'
    pred_json_path = 'large_jsons/predictions.json'
    tp_ids, fp_ids, fn_ids = measure_py_evaluator_time(gt_json_path, pred_json_path)
    measure_np_evaluator_time(gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    measure_jit_evaluator_time(gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    measure_taichi_evaluator_time(gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    check_evaluator_time(CyEvaluator, "CythonEvaluator", gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    check_evaluator_time(CyParallelEvaluator, "CyParallelEvaluator", gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    check_evaluator_time(CppEvaluator, "C++ Evaluator", gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    check_evaluator_time(ParallelCppEvaluator, "C++ Parallel Evaluator", gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)
    check_evaluator_time(SharedMutexParallelCppEvaluator, "C++ Parallel SharedMutex Evaluator", gt_json_path, pred_json_path, tp_ids, fp_ids, fn_ids)





