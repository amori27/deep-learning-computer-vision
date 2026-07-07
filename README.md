# Deep Learning & Computer Vision

Image classification (ResNet, custom CNNs) and object detection (YOLOv8, Faster R-CNN) with transfer learning, implemented in PyTorch and TensorFlow.

## Usage

```python
from src.classifier import ImageClassifier
classifier = ImageClassifier(model_name="resnet50")
prediction = classifier.predict("image.jpg")
```

```python
from src.detector import ObjectDetector
detector = ObjectDetector(model="yolov8")
boxes = detector.detect("scene.jpg")
```

## License

MIT
