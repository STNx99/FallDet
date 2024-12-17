import numpy as np


def output_to_keypoint(output, img_shape):
    """
    Convert model output to keypoints with rescaling to the original image shape.

    Args:
        output (numpy.ndarray): Model output with keypoint predictions.
        img_shape (tuple): Original image shape (height, width).

    Returns:
        list: Keypoints in format [(x1, y1, confidence1), (x2, y2, confidence2), ...].
    """
    keypoints = []
    img_h, img_w = img_shape

    for pred in output:
        kp = pred['keypoints']
        rescaled_kp = []

        for i in range(0, len(kp), 3):
            x, y, conf = kp[i], kp[i + 1], kp[i + 2]
            # Rescale to original image size
            x = int(x * img_w)
            y = int(y * img_h)
            rescaled_kp.append((x, y, conf))

        keypoints.append(rescaled_kp)

    return keypoints
