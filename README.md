# Python Performance Optimization using Numpy, JIT, Cython, TaiChi, C++, OpenMP

This project's goal is to highlight the importance of using the right tools
when the code is computationally intensive.\
The code is designed to address a computer vision task that finds the True
Positives (TP), False Positives (FP), and False Negatives (FN) in the
COCO dataset.

Here, we first implemented a basic code on `v1_python` that has `Evaluator` class which
reads two dataset in COCO format as ground truth json and prediction json and then finds
all True Positives (TP), False Positives (FP), False Negatives (FN) and returns their
corresponding annotation ids (in `evaluate` method).

Then we reimplement the same concept but utilizing tools that can give us performance
boost:

- Rewriting code with **numpy** operations.
- **JIT** optimizations
- Optimizing code with **tai_chi** framework
- **Cython** classes, methods,no `wraparound`, no `boundscheck`
- **Cython** `nogil` and parallel processing, arrays and `memoryviews`
- Python integration with **C++** and **pybind**
- C++ threading
- C++ **Threading** with **Shared Mutex**
- C++ **OpenMP** Optimization.

We tested the algorithms on 2 different sizes of jsons.

- Small size jsons: Almost **2K annotations**.
- Big size jsons: over **99K annotations**.

## JSONS with 2K annotations:

| version |              name               | process time (seconds) |      RANK #       |
|:-------:|:-------------------------------:|:----------------------:|:-----------------:|
|    8    |        **C++ Parallel**         |        0.00175         | :1st_place_medal: |
|   10    |     **C++ OpenMP Parallel**     |        0.01893         | :2nd_place_medal: |
|    9    | **C++ Parallel + Shared Mutex** |        0.00389         | :3rd_place_medal: |
|    7    |               C++               |        0.00451         |         4         |
|    5    |             Cython              |        0.05395         |         5         |
|    4    |             Tai-Chi             |         0.1000         |         6         |
|    6    |         Cython Parallel         |        0.13021         |         7         |
|    3    |               JIT               |         0.4453         |         8         |
|    1    |          basic python           |         0.9084         |         9         |

## JSONS with 100K annotations:

| version |              name               | process time (seconds) |      RANK #       |
|:-------:|:-------------------------------:|:----------------------:|:-----------------:|
|   10    |     **C++ OpenMP Parallel**     |        0.36951         | :1st_place_medal: |
|    8    |        **C++ Parallel**         |        0.42124         | :2nd_place_medal: |
|    9    | **C++ Parallel + Shared Mutex** |        1.28596         | :3rd_place_medal: |
|    7    |               C++               |        1.63921         |         4         |
|    4    |             Tai-Chi             |        2.47937         |         5         |
|    3    |               JIT               |        9.24433         |         6         |
|    5    |             Cython              |        19.06968        |         7         |
|    6    |         Cython Parallel         |        47.77951        |         8         |
|    1    |          basic python           |       322.95960        |         9         |

