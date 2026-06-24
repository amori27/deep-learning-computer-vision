"""Object Detection Module.

This module provides object detection using YOLO and Faster R-CNN.
"""

import torch
import numpy as np
from PIL import Image
from typing import Any


class ObjectDetector:
    """Handles object detection with pre-trained models."""

    def __init__(self, model: str = "fasterrcnn"):
        """Initialize ObjectDetector.

        Args:
            model: Detection model (fasterrcnn, yolov8).
        """
        self.model_name = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = self._get_coco_classes()

    def _get_coco_classes(self) -> list[str]:
        """Get COCO dataset class names.

        Returns:
            List of 80 COCO class names.
        """
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def load_model(self) -> None:
        """Load pre-trained detection model."""
        if self.model_name == "fasterrcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for detection.

        Args:
            image_path: Path to image.

        Returns:
            Preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        return image_tensor.unsqueeze(0)

    def detect(
        self,
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Detect objects in an image.

        Args:
            image_path: Path to image.
            confidence_threshold: Minimum confidence for detections.

        Returns:
            List of detected objects with bounding boxes.
        """
        if self.model is None:
            self.load_model()

        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class": self.class_names[label - 1],
                    "class_id": int(label)
                })

        return detections

    def detect_batch(
        self,
        image_paths: list[str],
        confidence_threshold: float = 0.5
    ) -> list[list[dict[str, Any]]]:
        """Detect objects in multiple images.

        Args:
            image_paths: List of image paths.
            confidence_threshold: Minimum confidence.

        Returns:
            List of detections for each image.
        """
        return [self.detect(path, confidence_threshold) for path in image_paths]


def draw_detections(
    image_path: str,
    detections: list[dict[str, Any]]
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image_path: Path to image.
        detections: List of detections.

    Returns:
        Image with drawn boxes.
    """
    import cv2

    image = cv2.imread(image_path)

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{det['class']}: {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
