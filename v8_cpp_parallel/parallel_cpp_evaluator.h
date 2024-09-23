#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include <mutex>
#include <thread>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

class BoundingBox {
public:
    int annotation_id;
    int image_id;
    int category_id;
    int x1, y1, x2, y2;
    int w, h;

    BoundingBox(int annotation_id, int image_id, int category_id, int x1, int y1, int w, int h)
        : annotation_id(annotation_id), image_id(image_id), category_id(category_id), x1(x1), y1(y1), w(w), h(h) {
        x2 = x1 + w;
        y2 = y1 + h;
    }

    double calculate_iou(const BoundingBox& other) const;

    bool is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, double iou_threshold) const;

    bool is_false_negative(const std::vector<BoundingBox>& predicted_boxes, double iou_threshold) const;
};

class ParallelCppEvaluator {
public:
    ParallelCppEvaluator(const std::string& ground_truth_json, const std::string& predictions_json);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> evaluate();

private:
    std::vector<BoundingBox> ground_truth_boxes;
    std::vector<BoundingBox> predicted_boxes;

    std::mutex mtx;

    void evaluate_boxes(int start, int end, std::vector<int>& tp_pred_ids, std::vector<int>& fp_pred_ids);
    void evaluate_ground_truths(int start, int end, std::vector<int>& fn_gt_ids);
};

#endif // EVALUATOR_H
