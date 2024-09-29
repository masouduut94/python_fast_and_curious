#include "parallel_cpp_evaluator.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

double BoundingBox::calculate_iou(const BoundingBox& other) const {
    int x1_inter = std::max(x1, other.x1);
    int y1_inter = std::max(y1, other.y1);
    int x2_inter = std::min(x2, other.x2);
    int y2_inter = std::min(y2, other.y2);

    int inter_area = std::max(0, x2_inter - x1_inter) * std::max(0, y2_inter - y1_inter);
    int box1_area = w * h;
    int box2_area = other.w * other.h;

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

ParallelCppEvaluator::ParallelCppEvaluator(const std::string& ground_truth_json, const std::string& predictions_json) {
    std::ifstream gt_file(ground_truth_json);
    std::ifstream pred_file(predictions_json);

    if (!gt_file.is_open() || !pred_file.is_open()) {
        throw std::runtime_error("Failed to open JSON files.");
    }

    json gt_data, pred_data;
    gt_file >> gt_data;
    pred_file >> pred_data;

    for (const auto& ann : gt_data["annotations"]) {
        ground_truth_boxes.emplace_back(ann["annotation_id"], ann["image_id"], ann["category_id"], ann["x1"], ann["y1"], ann["w"], ann["h"]);
    }
    for (const auto& ann : pred_data["annotations"]) {
        predicted_boxes.emplace_back(ann["annotation_id"], ann["image_id"], ann["category_id"], ann["x1"], ann["y1"], ann["w"], ann["h"]);
    }
}

void ParallelCppEvaluator::evaluate_boxes(int start, int end, std::vector<int>& tp_pred_ids, std::vector<int>& fp_pred_ids) {
    for (int i = start; i < end; ++i) {
        const auto& pred_box = predicted_boxes[i];
        bool is_tp = pred_box.is_true_positive_or_false_positive(ground_truth_boxes, 0.5);

        std::lock_guard<std::mutex> lock(mtx);
        if (is_tp) {
            tp_pred_ids.push_back(pred_box.annotation_id);
        } else {
            fp_pred_ids.push_back(pred_box.annotation_id);
        }
    }
}

void ParallelCppEvaluator::evaluate_ground_truths(int start, int end, std::vector<int>& fn_gt_ids) {
    for (int i = start; i < end; ++i) {
        const auto& gt_box = ground_truth_boxes[i];
        bool is_fn = gt_box.is_false_negative(predicted_boxes, 0.5);

        std::lock_guard<std::mutex> lock(mtx);
        if (is_fn) {
            fn_gt_ids.push_back(gt_box.annotation_id);
        }
    }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> ParallelCppEvaluator::evaluate() {
    std::vector<int> tp_pred_ids, fp_pred_ids, fn_gt_ids;

    int num_threads = std::thread::hardware_concurrency();
    int num_pred_boxes = predicted_boxes.size();
    int num_gt_boxes = ground_truth_boxes.size();

    std::vector<std::thread> threads;

    // Parallelize evaluation of predicted boxes
    for (int i = 0; i < num_threads; ++i) {
        int start = i * (num_pred_boxes / num_threads);
        int end = (i + 1) * (num_pred_boxes / num_threads);
        if (i == num_threads - 1) end = num_pred_boxes;
        threads.emplace_back(&ParallelCppEvaluator::evaluate_boxes, this, start, end, std::ref(tp_pred_ids), std::ref(fp_pred_ids));
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Parallelize evaluation of ground truth boxes
    for (int i = 0; i < num_threads; ++i) {
        int start = i * (num_gt_boxes / num_threads);
        int end = (i + 1) * (num_gt_boxes / num_threads);
        if (i == num_threads - 1) end = num_gt_boxes;
        threads.emplace_back(&ParallelCppEvaluator::evaluate_ground_truths, this, start, end, std::ref(fn_gt_ids));
    }
    for (auto& t : threads) t.join();

    return {tp_pred_ids, fp_pred_ids, fn_gt_ids};
}
