import json
import random


def create_test_data():
    ground_truth = {
        "annotations": []
    }
    predictions = {
        "annotations": []
    }

    image_id = 1
    category_id = 1
    annotation_id = 1

    # Generate True Positives
    for i in range(1500):
        x1 = random.randint(0, 500)
        y1 = random.randint(0, 500)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        gt_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "x1": x1,
            "y1": y1,
            "w": w,
            "h": h
        }
        ground_truth["annotations"].append(gt_box)

        pred_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "x1": x1 + random.randint(-5, 5),
            "y1": y1 + random.randint(-5, 5),
            "w": w,
            "h": h
        }

        predictions["annotations"].append(pred_box)
        annotation_id += 1

    # Generate False Positives
    for i in range(1500, 1800):
        x1 = random.randint(0, 500)
        y1 = random.randint(0, 500)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        pred_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "x1": x1,
            "y1": y1,
            "w": w,
            "h": h
        }
        predictions["annotations"].append(pred_box)

        annotation_id += 1

    # Generate False Negatives
    for i in range(1800, 2300):
        x1 = random.randint(0, 500)
        y1 = random.randint(0, 500)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        gt_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "x1": x1,
            "y1": y1,
            "w": w,
            "h": h
        }
        ground_truth["annotations"].append(gt_box)
        annotation_id += 1

    # Save data to JSON files
    with open('small_jsons/ground_truths.json', 'w') as gt_file:
        json.dump(ground_truth, gt_file, indent=2)

    with open('small_jsons/predictions.json', 'w') as preds_file:
        json.dump(predictions, preds_file, indent=2)


