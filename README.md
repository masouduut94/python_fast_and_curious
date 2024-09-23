# Python

This project consists of several implementations of COCO format dataset validation
between ground truths and predictions.\
It is designed to highlight the superpower of python libraries for big
computations as well as its friendship with other languages like C++.

Here, we first implemented a basic code on `v1_python` that has `Evaluator` class which
gets 2 paths of ground truth json and prediction json in COCO format. Then its `evaluate`
method finds all True Positives (TP), False Positives (FP), False Negatives (FN) and
corresponding annotation ids of the json files.

Since this procedure takes quite a lot of time (multiple for loops and IOU intensive
process), we investigate how we can improve the code performance by utilizing several
options:

- Rewriting code with numpy operations.
- JIT optimizations
- Optimizing code with tai_chi framework
- Cython classes, methods,no wraparound, no boundscheck
- Cython nogil and parallel processing, arrays and memoryviews
- Python integration with C++ and pybind
- C++ threading
- C++ Threading with Shared Mutex
- C++ OpenMP Optimization.

We tested the algorithms on 2 different sizes of jsons.

- Small size jsons: Almost 2K annotations.
- Big size jsons: over 99K annotations.

## JSONS with 2K annotations:

| version |            name             | process time (seconds) |      RANK #       |
|:-------:|:---------------------------:|:----------------------:|:-----------------:|
|    1    |        basic python         |         0.9084         |         9         |
|    2    |            numpy            |         2.5103         |        10         |
|    3    |             JIT             |         0.4453         |         8         |
|    4    |           Tai-Chi           |         0.1000         |         6         |
|    5    |           Cython            |        0.05395         |         5         |
|    6    |       Cython Parallel       |        0.13021         |         7         |
|    7    |             C++             |        0.00451         | :3rd_place_medal: |
|    8    |        C++ Parallel         |        0.00175         | :1st_place_medal: |
|    9    | C++ Parallel + Shared Mutex |        0.00389         | :2nd_place_medal: |
|   10    |     C++ OpenMP Parallel     |        0.01893         |         4         |

## JSONS with 99K annotations:

| version |            name             | process time (seconds) |      RANK #       |
|:-------:|:---------------------------:|:----------------------:|:-----------------:|
|    1    |        basic python         |          125           |         9         |
|    2    |            numpy            |          469           |        10         |
|    3    |             JIT             |         3.8314         |         6         |
|    4    |           Tai-Chi           |         2.2207         |         5         |
|    5    |           Cython            |        7.62609         |         7         |
|    6    |       Cython Parallel       |        18.65183        |         8         |
|    7    |             C++             |        0.63481         |         4         |
|    8    |        C++ Parallel         |        0.05023         | :1st_place_medal: |
|    9    | C++ Parallel + Shared Mutex |        0.53730         | :3rd_place_medal: |
|   10    |     C++ OpenMP Parallel     |        0.05228         | :2nd_place_medal: |

