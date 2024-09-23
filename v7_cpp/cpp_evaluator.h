#ifndef EVALUATOR_H  // Include guards to prevent double inclusion
#define EVALUATOR_H

#include <string>
#include <vector>

class BoundingBox {
public:
    // Constructor
    BoundingBox(int annotation_id, int image_id, int category_id, int x1, int y1, int w, int h);

    // Methods to determine true positives, false positives, etc.
    bool is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, float iou_threshold = 0.5) const;
    bool is_false_negative(const std::vector<BoundingBox>& predicted_boxes, float iou_threshold = 0.5) const;

    // Method to calculate IoU
    float calculate_iou(const BoundingBox& other_box) const;
    // Data members for bounding box attributes
    int annotation_id;
    int image_id;
    int category_id;
    int x1, y1, x2, y2, w, h;
};


class CppEvaluator {
public:
    CppEvaluator(const std::string& ground_truth_json, const std::string& predictions_json);
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> evaluate();
    std::vector<BoundingBox> ground_truth_boxes;
    std::vector<BoundingBox> predicted_boxes;
};

#endif  // EVALUATOR_H