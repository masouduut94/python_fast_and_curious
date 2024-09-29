#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <string>
#include <vector>
#include <shared_mutex> // For std::shared_mutex

class BoundingBox {
public:
    int annotation_id;
    int image_id;
    int category_id;
    int x1, y1, x2, y2;

    BoundingBox(int ann_id, int img_id, int cat_id, int x1, int y1, int w, int h);

    bool is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, double iou_threshold = 0.5) const;
    bool is_false_negative(const std::vector<BoundingBox>& predicted_boxes, double iou_threshold = 0.5) const;
    double calculate_iou(const BoundingBox& other_box) const;
};


class SharedMutexParallelCppEvaluator {
public:
    SharedMutexParallelCppEvaluator(const std::string& ground_truth_json, const std::string& predictions_json);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> evaluate();

private:
    std::vector<BoundingBox> ground_truth_boxes;
    std::vector<BoundingBox> predicted_boxes;
    std::shared_mutex mtx; // For concurrent access

    // Helper methods
    bool is_true_positive_or_false_positive(const BoundingBox& pred_box);
    bool is_false_negative(const BoundingBox& gt_box);
};

#endif // EVALUATOR_H
