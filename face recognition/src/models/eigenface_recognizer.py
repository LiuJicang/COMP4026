import json
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


@dataclass(frozen=True)
class Prediction:
    predicted_label: int | None
    predicted_name: str
    confidence: float
    is_unknown: bool


class EigenFaceRecognizerModel:
    """Thin wrapper around OpenCV EigenFaceRecognizer with manual unknown rejection."""

    def __init__(self, image_size: tuple[int, int], threshold: float, num_components: int = 0) -> None:
        self.image_size = tuple(int(v) for v in image_size)
        self.threshold = float(threshold)
        self.num_components = int(num_components)
        self.label_to_name: dict[int, str] = {}
        self._recognizer = cv2.face.EigenFaceRecognizer_create(
            self.num_components,
            float(sys.float_info.max),
        )

    def train(self, faces: list[np.ndarray], labels: list[int], label_to_name: dict[int, str]) -> None:
        if not faces:
            raise ValueError("No faces were provided for training.")

        processed_faces = [self._validate_face(face) for face in faces]
        label_array = np.asarray(labels, dtype=np.int32)
        self._recognizer.train(processed_faces, label_array)
        self.label_to_name = {int(key): str(value) for key, value in label_to_name.items()}

    def predict(self, face: np.ndarray) -> Prediction:
        prepared_face = self._validate_face(face)
        predicted_label, confidence = self._recognizer.predict(prepared_face)
        confidence = float(confidence)

        is_unknown = (
            predicted_label == -1
            or predicted_label not in self.label_to_name
            or confidence > self.threshold
        )

        if is_unknown:
            return Prediction(
                predicted_label=None,
                predicted_name="unknown",
                confidence=confidence,
                is_unknown=True,
            )

        return Prediction(
            predicted_label=int(predicted_label),
            predicted_name=self.label_to_name[int(predicted_label)],
            confidence=confidence,
            is_unknown=False,
        )

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._recognizer.write(str(output_dir / "eigenface_model.yml"))

        with (output_dir / "label_map.json").open("w", encoding="utf-8") as file:
            json.dump(self.label_to_name, file, indent=2, ensure_ascii=False)

        metadata = {
            "image_size": list(self.image_size),
            "threshold": self.threshold,
            "num_components": self.num_components,
        }
        with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, model_dir: str | Path) -> "EigenFaceRecognizerModel":
        model_dir = Path(model_dir)
        with (model_dir / "model_metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        model = cls(
            image_size=tuple(metadata["image_size"]),
            threshold=metadata["threshold"],
            num_components=metadata.get("num_components", 0),
        )
        model._recognizer.read(str(model_dir / "eigenface_model.yml"))

        with (model_dir / "label_map.json").open("r", encoding="utf-8") as file:
            raw_map = json.load(file)
        model.label_to_name = {int(key): str(value) for key, value in raw_map.items()}
        return model

    def _validate_face(self, face: np.ndarray) -> np.ndarray:
        if face.ndim != 2:
            raise ValueError("EigenFaceRecognizer expects grayscale 2D images.")
        if tuple(face.shape[::-1]) != self.image_size:
            raise ValueError(
                f"Unexpected face shape {face.shape}. Expected image size {self.image_size}."
            )
        return np.asarray(face, dtype=np.uint8)
