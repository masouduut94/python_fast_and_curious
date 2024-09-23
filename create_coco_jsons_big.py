import json
import random


def create_large_test_data():
    ground_truth = {
        "annotations": []
    }
    predictions = {
        "annotations": []
    }

    image_id = 1
    category_id = 1
    annotation_id = 1

    # Generate 85,000 True Positives (matching boxes in GT and PREDS)
    for i in range(85000):
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

        # Slightly adjust the predicted box to ensure it overlaps with the GT box
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

    # Generate 5,050 False Positives (boxes in PREDS but not in GT)
    for i in range(85000, 90050):
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

    # Generate 9,150 False Negatives (boxes in GT but not in PREDS)
    for i in range(90050, 99150):
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
    with open('large_jsons/ground_truths.json', 'w') as gt_file:
        json.dump(ground_truth, gt_file, indent=2)

    with open('large_jsons/predictions.json', 'w') as preds_file:
        json.dump(predictions, preds_file, indent=2)


# Run the function to generate the larger dataset
create_large_test_data()
