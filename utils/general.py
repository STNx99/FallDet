import numpy as np

def non_max_suppression_kpt(predictions, conf_thres=0.5, iou_thres=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on keypoints predictions.

    Args:
        predictions (list): List of keypoint predictions [(x, y, confidence), ...].
        conf_thres (float): Confidence threshold to filter low-confidence predictions.
        iou_thres (float): IoU threshold for suppression.

    Returns:
        list: Filtered keypoints after NMS.
    """
    def iou(box1, box2):
        # Calculate Intersection over Union (IoU) between two boxes
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    keep = []

    while predictions:
        best = predictions.pop(0)
        keep.append(best)

        predictions = [
            kp for kp in predictions
            if iou((best[0]-5, best[1]-5, best[0]+5, best[1]+5),  # Small bounding box around keypoint
                   (kp[0]-5, kp[1]-5, kp[0]+5, kp[1]+5)) < iou_thres
        ]

    # Filter low-confidence predictions
    return [kp for kp in keep if kp[2] > conf_thres]
