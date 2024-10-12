#include "cpp_evaluator.h"  // Include the header file
#include <fstream>
#include <nlohmann/json.hpp>  // Assuming you're using nlohmann JSON library

using json = nlohmann::json;

// BoundingBox constructor
BoundingBox::BoundingBox(int annotation_id, int image_id, int category_id, int x1, int y1, int w, int h)
    : annotation_id(annotation_id), image_id(image_id), category_id(category_id), x1(x1), y1(y1), w(w), h(h) {
    x2 = x1 + w;
    y2 = y1 + h;
}

// IoU calculation
float BoundingBox::calculate_iou(const BoundingBox& other_box) const {
    int x1_inter = std::max(this->x1, other_box.x1);
    int y1_inter = std::max(this->y1, other_box.y1);
    int x2_inter = std::min(this->x2, other_box.x2);
    int y2_inter = std::min(this->y2, other_box.y2);

    int inter_area = std::max(0, x2_inter - x1_inter) * std::max(0, y2_inter - y1_inter);
    int box1_area = this->w * this->h;
    int box2_area = other_box.w * other_box.h;

    return static_cast<float>(inter_area) / (box1_area + box2_area - inter_area);
}

bool BoundingBox::is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, float iou_threshold) const {
    for (const auto& gt_box : ground_truth_boxes) {
        if (gt_box.category_id == this->category_id && gt_box.image_id == this->image_id) {
            float iou = this->calculate_iou(gt_box);
            if (iou >= iou_threshold) {
                return true;
            }
        }
    }
    return false;
}

bool BoundingBox::is_false_negative(const std::vector<BoundingBox>& predicted_boxes, float iou_threshold) const {
    for (const auto& pred_box : predicted_boxes) {
        if (pred_box.category_id == this->category_id && pred_box.image_id == this->image_id) {
            float iou = this->calculate_iou(pred_box);
            if (iou >= iou_threshold) {
                return false;
            }
        }
    }
    return true;
}


// Evaluator constructor
CppEvaluator::CppEvaluator(const std::string& ground_truth_json, const std::string& predictions_json) {
    std::ifstream gt_file(ground_truth_json);
    std::ifstream pred_file(predictions_json);

    json gt_data, pred_data;
    gt_file >> gt_data;
    pred_file >> pred_data;

    for (const auto& ann : gt_data["annotations"]) {
        ground_truth_boxes.emplace_back(
            ann["annotation_id"], ann["image_id"], ann["category_id"],
            ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]);
    }

    for (const auto& ann : pred_data["annotations"]) {
        predicted_boxes.emplace_back(
            ann["annotation_id"], ann["image_id"], ann["category_id"],
            ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]);
    }
}

// Implementation of evaluate
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> CppEvaluator::evaluate() {
    std::vector<int> tp_pred_ids, fp_pred_ids, fn_gt_ids;

    for (const auto& pred_box : predicted_boxes) {
        bool is_tp = pred_box.is_true_positive_or_false_positive(ground_truth_boxes);
        if (is_tp) {
            tp_pred_ids.push_back(pred_box.annotation_id);
        } else {
            fp_pred_ids.push_back(pred_box.annotation_id);
        }
    }

    for (const auto& gt_box : ground_truth_boxes) {
        bool is_fn = gt_box.is_false_negative(predicted_boxes);
        if (is_fn) {
            fn_gt_ids.push_back(gt_box.annotation_id);
        }
    }

    return {tp_pred_ids, fp_pred_ids, fn_gt_ids};
}
