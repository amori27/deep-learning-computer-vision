# Deep Learning & Computer Vision

Image classification (ResNet, custom CNNs) and object detection (YOLOv8, Faster R-CNN) with transfer learning, implemented in PyTorch and TensorFlow.

> **Before you install:** this project depends on both PyTorch and TensorFlow, and a default `pip install -r requirements.txt` can end up fetching gigabytes of CUDA libraries. On a CPU-only machine, install the CPU wheels first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then install the rest of `requirements.txt`. The included tests are simple smoke tests and mock the heavy parts, so `uv run --with pytest pytest tests/ -q` verifies the repository in seconds without downloading any of the DL stack.

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
