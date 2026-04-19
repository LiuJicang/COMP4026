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


class BaseFaceRecognizerModel:
    """Common wrapper logic for OpenCV face recognizers with manual unknown rejection."""

    model_name = "base"
    model_filename = "model.yml"

    def __init__(self, image_size: tuple[int, int], threshold: float) -> None:
        self.image_size = tuple(int(v) for v in image_size)
        self.threshold = float(threshold)
        self.label_to_name: dict[int, str] = {}
        self._recognizer = self._create_recognizer()

    def _create_recognizer(self):
        raise NotImplementedError

    def _extra_metadata(self) -> dict:
        return {}

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

        self._recognizer.write(str(output_dir / self.model_filename))

        with (output_dir / "label_map.json").open("w", encoding="utf-8") as file:
            json.dump(self.label_to_name, file, indent=2, ensure_ascii=False)

        metadata = {
            "model_name": self.model_name,
            "image_size": list(self.image_size),
            "threshold": self.threshold,
            **self._extra_metadata(),
        }
        with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    def _validate_face(self, face: np.ndarray) -> np.ndarray:
        if face.ndim != 2:
            raise ValueError(f"{self.__class__.__name__} expects grayscale 2D images.")
        if tuple(face.shape[::-1]) != self.image_size:
            raise ValueError(
                f"Unexpected face shape {face.shape}. Expected image size {self.image_size}."
            )
        return np.asarray(face, dtype=np.uint8)


class EigenFaceRecognizerModel(BaseFaceRecognizerModel):
    """Thin wrapper around OpenCV EigenFaceRecognizer with manual unknown rejection."""

    model_name = "eigenfaces"
    model_filename = "eigenface_model.yml"

    def __init__(self, image_size: tuple[int, int], threshold: float, num_components: int = 0) -> None:
        self.num_components = int(num_components)
        super().__init__(image_size=image_size, threshold=threshold)

    def _create_recognizer(self):
        return cv2.face.EigenFaceRecognizer_create(
            self.num_components,
            float(sys.float_info.max),
        )

    def _extra_metadata(self) -> dict:
        return {"num_components": self.num_components}

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
        model._recognizer.read(str(model_dir / cls.model_filename))

        with (model_dir / "label_map.json").open("r", encoding="utf-8") as file:
            raw_map = json.load(file)
        model.label_to_name = {int(key): str(value) for key, value in raw_map.items()}
        return model


class LBPHFaceRecognizerModel(BaseFaceRecognizerModel):
    """Wrapper around OpenCV LBPHFaceRecognizer."""

    model_name = "lbph"
    model_filename = "lbph_model.yml"

    def __init__(
        self,
        image_size: tuple[int, int],
        threshold: float,
        radius: int = 1,
        neighbors: int = 8,
        grid_x: int = 8,
        grid_y: int = 8,
    ) -> None:
        self.radius = int(radius)
        self.neighbors = int(neighbors)
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)
        super().__init__(image_size=image_size, threshold=threshold)

    def _create_recognizer(self):
        return cv2.face.LBPHFaceRecognizer_create(
            self.radius,
            self.neighbors,
            self.grid_x,
            self.grid_y,
            float(sys.float_info.max),
        )

    def _extra_metadata(self) -> dict:
        return {
            "radius": self.radius,
            "neighbors": self.neighbors,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
        }

    @classmethod
    def load(cls, model_dir: str | Path) -> "LBPHFaceRecognizerModel":
        model_dir = Path(model_dir)
        with (model_dir / "model_metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        model = cls(
            image_size=tuple(metadata["image_size"]),
            threshold=metadata["threshold"],
            radius=metadata.get("radius", 1),
            neighbors=metadata.get("neighbors", 8),
            grid_x=metadata.get("grid_x", 8),
            grid_y=metadata.get("grid_y", 8),
        )
        model._recognizer.read(str(model_dir / cls.model_filename))

        with (model_dir / "label_map.json").open("r", encoding="utf-8") as file:
            raw_map = json.load(file)
        model.label_to_name = {int(key): str(value) for key, value in raw_map.items()}
        return model


def build_face_recognizer_from_config(cfg: dict) -> BaseFaceRecognizerModel:
    model_cfg = cfg["model"]
    recognizer_name = model_cfg.get("recognizer", "eigenfaces").lower()
    image_size = tuple(cfg["preprocess"]["image_size"])
    threshold = model_cfg["threshold"]

    if recognizer_name == "eigenfaces":
        return EigenFaceRecognizerModel(
            image_size=image_size,
            threshold=threshold,
            num_components=model_cfg.get("num_components", 0),
        )

    if recognizer_name == "lbph":
        return LBPHFaceRecognizerModel(
            image_size=image_size,
            threshold=threshold,
            radius=model_cfg.get("radius", 1),
            neighbors=model_cfg.get("neighbors", 8),
            grid_x=model_cfg.get("grid_x", 8),
            grid_y=model_cfg.get("grid_y", 8),
        )

    raise ValueError(f"Unsupported recognizer: {recognizer_name}")


def load_face_recognizer_from_dir(model_dir: str | Path) -> BaseFaceRecognizerModel:
    model_dir = Path(model_dir)
    with (model_dir / "model_metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    recognizer_name = metadata.get("model_name", "eigenfaces").lower()
    if recognizer_name == "eigenfaces":
        return EigenFaceRecognizerModel.load(model_dir)
    if recognizer_name == "lbph":
        return LBPHFaceRecognizerModel.load(model_dir)
    raise ValueError(f"Unsupported saved recognizer: {recognizer_name}")
