"""
models_tfhub.py
----------------
A lightweight wrapper for TensorFlow Hub object detection models.
Lets you easily swap between pretrained detectors like:
 - SSD MobileNet V2
 - EfficientDet D0 / D1

Returns normalized boxes, class IDs, and confidence scores.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

# Map short names to TensorFlow Hub handles
# (feel free to add EfficientDet D2/D3 later)
MODEL_HANDLES = {
    "ssd_mobilenet_v2": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
    "efficientdet_d0":  "https://tfhub.dev/tensorflow/efficientdet/d0/1",
    "efficientdet_d1":  "https://tfhub.dev/tensorflow/efficientdet/d1/1",
}

class TFHubDetector:
    """
    Wrapper for TF Hub object-detection models.

    Example:
        >>> from models_tfhub import TFHubDetector
        >>> import cv2
        >>> det = TFHubDetector("ssd_mobilenet_v2")
        >>> img = cv2.imread("example.jpg")
        >>> boxes, scores, classes, dt = det(img)
    """

    def __init__(self, model_name="ssd_mobilenet_v2",
                 score_thresh=0.4, max_dets=100):
        if model_name not in MODEL_HANDLES:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Choices: {list(MODEL_HANDLES.keys())}")

        print(f"[INFO] Loading TF-Hub model: {model_name}")
        self.handle = MODEL_HANDLES[model_name]
        self.model = hub.load(self.handle)
        self.infer = self.model.signatures["serving_default"]

        self.score_thresh = score_thresh
        self.max_dets = max_dets


    def __call__(self, image_bgr):
        """
        Run inference on a single OpenCV (BGR) image.

        Args:
            image_bgr (np.ndarray): HxWx3 uint8

        Returns:
            boxes   (Nx4) np.ndarray — normalized [y1, x1, y2, x2]
            scores  (N,)   np.ndarray — confidence scores
            classes (N,)   np.ndarray — integer class IDs
            dt       (float) — inference time in seconds
        """
        # Convert BGR → RGB and tensorize
        rgb = image_bgr[:, :, ::-1]
        rgb = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        rgb = tf.expand_dims(rgb, 0)

        t0 = time.time()
        outputs = self.infer(rgb)
        dt = time.time() - t0

        # Extract model outputs
        boxes   = outputs["detection_boxes"].numpy()[0]      # [N, y1,x1,y2,x2]
        scores  = outputs["detection_scores"].numpy()[0]
        classes = outputs["detection_classes"].numpy()[0].astype(np.int32)

        # Filter by confidence
        keep = scores >= self.score_thresh
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        # Keep only top K detections
        if len(scores) > self.max_dets:
            idx = np.argsort(-scores)[:self.max_dets]
            boxes, scores, classes = boxes[idx], scores[idx], classes[idx]

        return boxes, scores, classes, dt
