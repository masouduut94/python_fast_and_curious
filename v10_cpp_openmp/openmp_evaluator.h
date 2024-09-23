#ifndef OPENMP_EVALUATOR_H
#define OPENMP_EVALUATOR_H

#include <vector>
#include <string>
#include <tuple>

// BoundingBox class declaration
class BoundingBox {
public:
    // Constructor
    BoundingBox(int ann_id, int img_id, int cat_id, int x1, int y1, int w, int h);

    // Method to calculate Intersection over Union (IoU) with another bounding box
    double calculate_iou(const BoundingBox& other_box) const;

    // Check if the box is a true positive or false positive based on ground truth boxes
    bool is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, double iou_threshold = 0.5) const;

    // Check if the box is a false negative based on predicted boxes
    bool is_false_negative(const std::vector<BoundingBox>& predicted_boxes, double iou_threshold = 0.5) const;

    // Public member variables
    int annotation_id;
    int image_id;
    int category_id;
    int x1, y1, x2, y2;  // Coordinates of the bounding box (top-left and bottom-right)
};

// OpenmpEvaluator class declaration
class OpenmpEvaluator {
public:
    // Constructor that loads ground truth and predicted bounding boxes from JSON files
    OpenmpEvaluator(const std::string& ground_truth_json, const std::string& predictions_json);

    // Evaluate the predictions, returning true positives, false positives, and false negatives
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> evaluate();

private:
    std::vector<BoundingBox> ground_truth_boxes;   // Ground truth bounding boxes
    std::vector<BoundingBox> predicted_boxes;      // Predicted bounding boxes
};

#endif // OPENMP_EVALUATOR_H
