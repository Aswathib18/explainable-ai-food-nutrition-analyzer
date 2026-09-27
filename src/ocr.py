"""
Optical Character Recognition (OCR) Engine for Food Package Scanning.

Employs local, open-source EasyOCR with deep learning text detection (CRAFT)
and CRNN recognition to extract text, bounding coordinates, and confidence scores
from captured food packaging images.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import logging

from src.image_processing import load_image, pil_to_cv2

logger = logging.getLogger(__name__)

# Global singleton OCR reader instance for reuse across requests
_OCR_READER = None


def get_ocr_reader():
    """
    Lazy-initialization of EasyOCR Reader.
    Runs locally on CPU for universal cross-platform compatibility without cloud APIs.
    """
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (CPU mode)...")
            _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized successfully.")
        except Exception as exc:
            logger.error(f"Failed to initialize EasyOCR: {exc}")
            _OCR_READER = None
    return _OCR_READER


def run_ocr(
    image_input: Union[Image.Image, np.ndarray, str]
) -> Dict[str, Any]:
    """
    Perform real OCR on captured food label image.

    Returns:
        {
            "success": bool,
            "raw_text": str,
            "confidence": float (0.0 - 1.0),
            "detected_regions": [
                {"text": "...", "confidence": 0.95, "box": [[x,y], ...]}
            ],
            "line_count": int,
            "error_message": Optional[str]
        }
    """
    try:
        # Normalize input to numpy array
        if isinstance(image_input, Image.Image):
            pil_img = load_image(image_input)
            img_np = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            pil_img = load_image(image_input)
            img_np = np.array(pil_img)

        reader = get_ocr_reader()
        if reader is None:
            return {
                "success": False,
                "raw_text": "",
                "confidence": 0.0,
                "detected_regions": [],
                "line_count": 0,
                "error_message": "OCR engine could not be initialized in this environment."
            }

        # Run OCR detection & recognition
        # detail=1 returns list of (bbox, text, confidence)
        results = reader.readtext(img_np, detail=1, paragraph=False)

        if not results:
            return {
                "success": False,
                "raw_text": "",
                "confidence": 0.0,
                "detected_regions": [],
                "line_count": 0,
                "error_message": "Unable to reliably read the label. No text regions identified."
            }

        detected_regions: List[Dict[str, Any]] = []
        text_lines: List[str] = []
        confidences: List[float] = []

        for bbox, text, conf in results:
            clean_text = str(text).strip()
            if clean_text:
                conf_val = float(conf)
                # Convert bbox points to serializable float lists
                box_coords = [[float(pt[0]), float(pt[1])] for pt in bbox]
                detected_regions.append({
                    "text": clean_text,
                    "confidence": round(conf_val, 3),
                    "box": box_coords
                })
                text_lines.append(clean_text)
                confidences.append(conf_val)

        if not text_lines:
            return {
                "success": False,
                "raw_text": "",
                "confidence": 0.0,
                "detected_regions": [],
                "line_count": 0,
                "error_message": "Unable to reliably read the label. No legible text found."
            }

        raw_text = "\n".join(text_lines)
        avg_confidence = round(float(np.mean(confidences)), 3)

        return {
            "success": True,
            "raw_text": raw_text,
            "confidence": avg_confidence,
            "detected_regions": detected_regions,
            "line_count": len(text_lines),
            "error_message": None
        }

    except Exception as exc:
        logger.error(f"OCR execution error: {exc}")
        return {
            "success": False,
            "raw_text": "",
            "confidence": 0.0,
            "detected_regions": [],
            "line_count": 0,
            "error_message": f"Unable to reliably read the label: {str(exc)}"
        }
