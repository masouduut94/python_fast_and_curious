import time
# from v10_cpp_openmp.test import gt_json
from v1_python.py_evaluator import Evaluator
from v2_numpy.np_evaluator import Evaluator as NpEvaluator
from v3_jit_functional.jit_evaluator import measure_jit_evaluator_time
from v4_tai_chi.taichi_evaluator import measure_taichi_evaluator_time
from v5_cython.evaluator import Evaluator as CyEvaluator
from v6_parallel_cython.evaluator import Evaluator as CyParallelEvaluator
from v7_cpp.cpp_evaluator import CppEvaluator
from v8_cpp_parallel.parallel_cpp_evaluator import ParallelCppEvaluator
from v9_cpp_parallel_shared_mutex.shared_mutex_parallel_evaluator import SharedMutexParallelCppEvaluator
from v10_cpp_openmp.openmp_evaluator import OpenmpEvaluator


def check_evaluator_time(module, gt_json_path, pred_json_path):
    evaluator = module(gt_json_path, pred_json_path)
    t1 = time.time()
    tp_ids, fp_ids, fn_ids = evaluator.evaluate()
    t2 = time.time()
    return t2 - t1, sorted(tp_ids), sorted(fp_ids), sorted(fn_ids)


if __name__ == "__main__":
    gt_json_path = 'small_jsons/ground_truths.json'
    pred_json_path = 'small_jsons/predictions.json'
    t0, tp_ids, fp_ids, fn_ids = check_evaluator_time(Evaluator, gt_json_path, pred_json_path)
    t1, tp1, fp1, fn1 = check_evaluator_time(NpEvaluator, gt_json_path, pred_json_path)
    t2, tp2, fp2, fn2 = measure_jit_evaluator_time(gt_json_path, pred_json_path)
    t3, tp3, fp3, fn3 = measure_taichi_evaluator_time(gt_json_path, pred_json_path)
    t4, tp4, fp4, fn4 = check_evaluator_time(CyEvaluator, gt_json_path, pred_json_path)
    t5, tp5, fp5, fn5 = check_evaluator_time(CyParallelEvaluator, gt_json_path, pred_json_path)
    t6, tp6, fp6, fn6 = check_evaluator_time(CppEvaluator, gt_json_path, pred_json_path)
    t7, tp7, fp7, fn7 = check_evaluator_time(ParallelCppEvaluator, gt_json_path, pred_json_path)
    t8, tp8, fp8, fn8 = check_evaluator_time(SharedMutexParallelCppEvaluator, gt_json_path, pred_json_path)
    t9, tp9, fp9, fn9 = check_evaluator_time(OpenmpEvaluator, gt_json_path, pred_json_path)

    assert tp_ids == tp1 == tp2 == tp3 == tp4 == tp5 == tp6 == tp7 == tp8 == tp9
    assert fp_ids == fp1 == fp2 == fp3 == fp4 == fp5 == fp6 == fp7 == fp8 == fp9
    assert fn_ids == fn1 == fn2 == fn3 == fn4 == fn5 == fn6 == fn7 == fn8 == fn9

    p = [
        ('Simple Python', t0),
        ('Numpy', t1),
        ('JIT',  t2),
        ('TaiChi', t3),
        ('Basic Cython', t4),
        ('Parallel Cython', t5),
        ('Simple C++', t6),
        ('Parallel C++', t7),
        ('Parallel C++ + shared mutex', t8),
        ('C++ + OpenMP ', t9)
    ]

    p = sorted(p, key=lambda x: x[1])
    for i, (name, execution_time) in enumerate(p):
        print(f"Rank #{i+1}, {name} | {execution_time: .5f} seconds")
