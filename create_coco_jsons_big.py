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

    # Target values for proportions
    n1 = 80000
    n2 = 20000
    n3 = 2000

    for i in range(n1):
        x1 = random.randint(0, 500)
        y1 = random.randint(0, 500)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        # Ground truth box (matching GT and prediction for TP)
        gt_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h]
        }
        ground_truth["annotations"].append(gt_box)

        # Prediction box (slightly varied to simulate IoU)
        pred_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            'bbox': [
                x1 + random.randint(-5, 5),
                y1 + random.randint(-5, 5),
                w,
                h
            ],
        }
        predictions["annotations"].append(pred_box)

        annotation_id += 1

    for i in range(n2):
        x1 = random.randint(0, 600)
        y1 = random.randint(0, 600)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        pred_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h],
        }
        predictions["annotations"].append(pred_box)

        annotation_id += 1

    # Generate False Negatives (boxes in GT with no corresponding prediction)
    for i in range(n3):
        x1 = random.randint(0, 1000)
        y1 = random.randint(0, 1000)
        w = random.randint(20, 70)
        h = random.randint(20, 70)

        # Ground truth box with no prediction match
        gt_box = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h],
        }
        ground_truth["annotations"].append(gt_box)

        annotation_id += 1

    # Save data to JSON files
    with open('large_jsons/ground_truths.json', 'w') as gt_file:
        json.dump(ground_truth, gt_file, indent=2)

    with open('large_jsons/predictions.json', 'w') as preds_file:
        json.dump(predictions, preds_file, indent=2)


if __name__ == '__main__':
    
    # Run the function to generate the larger dataset
    create_test_data()
