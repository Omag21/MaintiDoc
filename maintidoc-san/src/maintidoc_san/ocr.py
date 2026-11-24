"""OCR local (RapidOCR, modèles ONNX embarqués) : texte des schémas et des pages scannées.

Aucune donnée ne quitte la machine. Le moteur est chargé à la première utilisation ;
une interface commune (`OCREngine`) permet de le remplacer par un faux dans les tests.
"""
from __future__ import annotations

from typing import Protocol

MIN_CONFIDENCE = 0.6  # lignes moins sûres ignorées (bruit, fragments de traits)


class OCREngine(Protocol):
    def read(self, image: bytes) -> list[str]:
        """Renvoie les lignes de texte lues dans l'image (PNG/JPEG en octets)."""
        ...


class RapidOCREngine:
    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        self.min_confidence = min_confidence
        self._engine = None

    def read(self, image: bytes) -> list[str]:
        if self._engine is None:
            from rapidocr import RapidOCR  # import paresseux : dépendance lourde (OpenCV, ONNX)
            self._engine = RapidOCR()
        result = self._engine(image)
        if not result.txts:
            return []
        return [text.strip() for text, score in zip(result.txts, result.scores)
                if score >= self.min_confidence and text.strip()]