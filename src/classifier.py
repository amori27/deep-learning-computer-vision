"""Image Classification Module.

This module provides image classification using pre-trained
CNN models including ResNet, VGG, and EfficientNet.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Any


class ImageClassifier:
    """Handles image classification with pre-trained models."""

    def __init__(self, model_name: str = "resnet50"):
        """Initialize the ImageClassifier.

        Args:
            model_name: Name of the model (resnet50, vgg16, efficientnet).
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = self._load_class_names()
        self.transform = self._get_transform()

    def _load_class_names(self) -> list[str]:
        """Load ImageNet class names.

        Returns:
            List of 1000 ImageNet class names.
        """
        return [f"class_{i}" for i in range(1000)]

    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline.

        Returns:
            Transform compose object.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self) -> None:
        """Load the pre-trained model."""
        if self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
        elif self.model_name == "vgg16":
            self.model = torchvision.models.vgg16(pretrained=True)
        elif self.model_name == "efficientnet":
            self.model = torchvision.models.efficientnet_b0(pretrained=True)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for classification.

        Args:
            image_path: Path to image file.

        Returns:
            Preprocessed tensor.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def predict(self, image_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict class probabilities for an image.

        Args:
            image_path: Path to image.
            top_k: Number of top predictions.

        Returns:
            List of (class_name, probability) tuples.
        """
        if self.model is None:
            self.load_model()

        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append((self.class_names[idx.item()], prob.item()))

        return results

    def classify_batch(self, image_paths: list[str]) -> list[list[tuple[str, float]]]:
        """Classify multiple images.

        Args:
            image_paths: List of image paths.

        Returns:
            List of predictions for each image.
        """
        return [self.predict(path) for path in image_paths]


class FeatureExtractor:
    """Extract features from images using pre-trained models."""

    def __init__(self, model_name: str = "resnet50"):
        """Initialize FeatureExtractor.

        Args:
            model_name: Model name for feature extraction.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self) -> None:
        """Load model and remove classification head."""
        if self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract(self, image_path: str) -> list[float]:
        """Extract features from an image.

        Args:
            image_path: Path to image.

        Returns:
            Feature vector.
        """
        if self.model is None:
            self.load_model()

        classifier = ImageClassifier(self.model_name)
        image_tensor = classifier.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.cpu().numpy()[0].tolist()
