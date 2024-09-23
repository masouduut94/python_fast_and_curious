#include "openmp_evaluator.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>  // For OpenMP

using json = nlohmann::json;

// BoundingBox class implementation
BoundingBox::BoundingBox(int ann_id, int img_id, int cat_id, int x1, int y1, int w, int h)
    : annotation_id(ann_id), image_id(img_id), category_id(cat_id), x1(x1), y1(y1), x2(x1 + w), y2(y1 + h) {}

double BoundingBox::calculate_iou(const BoundingBox& other_box) const {
    int x1_inter = std::max(x1, other_box.x1);
    int y1_inter = std::max(y1, other_box.y1);
    int x2_inter = std::min(x2, other_box.x2);
    int y2_inter = std::min(y2, other_box.y2);

    int inter_area = std::max(0, x2_inter - x1_inter) * std::max(0, y2_inter - y1_inter);

    int box1_area = (x2 - x1) * (y2 - y1);
    int box2_area = (other_box.x2 - other_box.x1) * (other_box.y2 - other_box.y1);

    return static_cast<double>(inter_area) / (box1_area + box2_area - inter_area);
}

bool BoundingBox::is_true_positive_or_false_positive(const std::vector<BoundingBox>& ground_truth_boxes, double iou_threshold) const {
    for (const auto& gt_box : ground_truth_boxes) {
        if (gt_box.category_id == category_id && gt_box.image_id == image_id) {
            double iou = this->calculate_iou(gt_box);
            if (iou >= iou_threshold) {
                return true;
            }
        }
    }
    return false;
}

bool BoundingBox::is_false_negative(const std::vector<BoundingBox>& predicted_boxes, double iou_threshold) const {
    for (const auto& pred_box : predicted_boxes) {
        if (pred_box.category_id == category_id && pred_box.image_id == image_id) {
            double iou = this->calculate_iou(pred_box);
            if (iou >= iou_threshold) {
                return false;
            }
        }
    }
    return true;
}

// OpenmpEvaluator class implementation
OpenmpEvaluator::OpenmpEvaluator(const std::string& ground_truth_json, const std::string& predictions_json) {
    std::ifstream gt_file(ground_truth_json);
    std::ifstream pred_file(predictions_json);

    if (!gt_file.is_open() || !pred_file.is_open()) {
        throw std::runtime_error("Error opening JSON files.");
    }

    json gt_data;
    json pred_data;

    gt_file >> gt_data;
    pred_file >> pred_data;

    for (const auto& ann : gt_data["annotations"]) {
        ground_truth_boxes.emplace_back(
            ann["annotation_id"], ann["image_id"], ann["category_id"],
            ann["x1"], ann["y1"], ann["w"], ann["h"]
        );
    }

    for (const auto& ann : pred_data["annotations"]) {
        predicted_boxes.emplace_back(
            ann["annotation_id"], ann["image_id"], ann["category_id"],
            ann["x1"], ann["y1"], ann["w"], ann["h"]
        );
    }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> OpenmpEvaluator::evaluate() {
    std::vector<int> tp_pred_ids;
    std::vector<int> fp_pred_ids;
    std::vector<int> fn_gt_ids;

    // Parallel loop for predictions
    #pragma omp parallel for
    for (int i = 0; i < predicted_boxes.size(); ++i) {
        const auto& pred_box = predicted_boxes[i];
        if (pred_box.is_true_positive_or_false_positive(ground_truth_boxes)) {
            #pragma omp critical
            tp_pred_ids.push_back(pred_box.annotation_id);
        } else {
            #pragma omp critical
            fp_pred_ids.push_back(pred_box.annotation_id);
        }
    }

    // Parallel loop for ground truth
    #pragma omp parallel for
    for (int i = 0; i < ground_truth_boxes.size(); ++i) {
        const auto& gt_box = ground_truth_boxes[i];
        if (gt_box.is_false_negative(predicted_boxes)) {
            #pragma omp critical
            fn_gt_ids.push_back(gt_box.annotation_id);
        }
    }

    return std::make_tuple(tp_pred_ids, fp_pred_ids, fn_gt_ids);
}
