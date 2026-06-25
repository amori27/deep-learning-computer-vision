# Deep Learning Computer Vision
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Computer vision toolkit featuring image classification, object detection, and pre-trained model implementations using PyTorch and TensorFlow.

## Description

Production-ready deep learning models for computer vision tasks including image classification with CNNs, object detection with YOLO and Faster R-CNN, and transfer learning from ImageNet pre-trained models.

## Skills & Technologies

- Python 3.9+
- PyTorch
- TensorFlow
- OpenCV
- NumPy
- torchvision
- CNN Architectures
- Transfer Learning
- Object Detection

## Installation

```bash
git clone https://github.com/amori27/deep-learning-computer-vision.git
cd deep-learning-computer-vision
pip install -r requirements.txt
```

## Usage

### Image Classification

```python
from src.classifier import ImageClassifier

classifier = ImageClassifier(model_name="resnet50")
prediction = classifier.predict("image.jpg")
```

### Object Detection

```python
from src.detector import ObjectDetector

detector = ObjectDetector(model="yolov8")
boxes = detector.detect("scene.jpg")
```

## Project Structure

```
deep-learning-computer-vision/
├── src/
│   ├── classifier.py       # Image classification
│   ├── detector.py         # Object detection
│   ├── models.py           # Model definitions
│   └── transforms.py        # Image transforms
├── requirements.txt
└── README.md
```

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

MIT License
