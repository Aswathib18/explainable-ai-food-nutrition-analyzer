"""
Image preprocessing and computer vision transformation utilities.
Handles image loading, color-space conversion, resizing, and normalization
for deep learning inference and optical character recognition (OCR) pipelines.

All cv2 operations have pure PIL / NumPy fallbacks so the module works on
Streamlit Cloud (headless Linux) even when OpenCV is unavailable.
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

import config


# ---------------------------------------------------------------------------
# PIL-only helpers (always available, no cv2 dependency)
# ---------------------------------------------------------------------------

def _pil_to_gray_np(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to a uint8 grayscale NumPy array (H, W)."""
    return np.array(image.convert("L"), dtype=np.uint8)


def _gray_np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert uint8 grayscale NumPy array to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _pil_clahe_approx(gray_np: np.ndarray) -> np.ndarray:
    """Approximate CLAHE via PIL histogram equalisation."""
    return np.array(ImageOps.equalize(_gray_np_to_pil(gray_np)), dtype=np.uint8)


def _pil_bilateral_approx(gray_np: np.ndarray) -> np.ndarray:
    """Approximate bilateral filter using a gentle Gaussian smooth via PIL."""
    return np.array(_gray_np_to_pil(gray_np).filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.uint8)


def _pil_sharpen(gray_np: np.ndarray) -> np.ndarray:
    """Sharpen using PIL built-in SHARPEN kernel."""
    return np.array(_gray_np_to_pil(gray_np).filter(ImageFilter.SHARPEN), dtype=np.uint8)


def _pil_otsu_threshold(gray_np: np.ndarray) -> np.ndarray:
    """Otsu-style global binarisation implemented with NumPy only. Returns uint8 (0/255)."""
    hist, _ = np.histogram(gray_np.ravel(), bins=256, range=(0, 256))
    total = gray_np.size
    best_thresh = 0
    best_var = 0.0
    weight_bg = 0.0
    sum_bg = 0.0
    total_sum = float(np.dot(np.arange(256, dtype=np.float64), hist))
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (total_sum - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t
    return np.where(gray_np > best_thresh, 255, 0).astype(np.uint8)


def _pil_adaptive_threshold(gray_np: np.ndarray, block_size: int = 21, c: int = 10) -> np.ndarray:
    """Gaussian adaptive threshold via PIL Gaussian blur. Returns uint8 (0/255)."""
    blurred = np.array(
        _gray_np_to_pil(gray_np).filter(ImageFilter.GaussianBlur(radius=block_size // 2)),
        dtype=np.int32
    )
    return np.where(gray_np.astype(np.int32) > blurred - c, 255, 0).astype(np.uint8)


def _laplacian_variance(gray_np: np.ndarray) -> float:
    """Estimate sharpness via discrete Laplacian using pure NumPy."""
    h, w = gray_np.shape
    if h < 3 or w < 3:
        return 0.0
    gray_f = gray_np.astype(np.float32)
    lap = (
        gray_f[:-2, 1:-1] + gray_f[2:, 1:-1]
        + gray_f[1:-1, :-2] + gray_f[1:-1, 2:]
        - 4 * gray_f[1:-1, 1:-1]
    )
    return float(np.var(lap))


# ==============================================================================
# PUBLIC API
# ==============================================================================

def load_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """
    Load an image from various input types (file path, bytes, or existing PIL Image)
    and ensure it is converted to RGB.
    """
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input))
    else:
        # Handles bytes or BytesIO stream
        image = Image.open(image_input)

    # Correct orientation based on EXIF tags (e.g. mobile photo rotation)
    image = ImageOps.exif_transpose(image)

    # Ensure 3-channel RGB format (stripping alpha channel if PNG)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = config.IMAGE_INPUT_SIZE
) -> np.ndarray:
    """
    Preprocess image for standard MobileNetV2 ONNX deep learning classification.
    Steps:
    1. Resize image with bilinear/bicubic resampling.
    2. Convert PIL Image to float32 NumPy array in range [0.0, 1.0].
    3. Normalize using standard ImageNet mean and standard deviation:
       normalized = (x - mean) / std
    4. Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width).
    5. Add batch dimension -> Shape: (1, 3, 224, 224).
    """
    resized = image.resize(target_size, Image.Resampling.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0

    # Apply ImageNet standardization
    mean = np.array(config.IMAGE_MEAN, dtype=np.float32)
    std = np.array(config.IMAGE_STD, dtype=np.float32)
    normalized = (img_np - mean) / std

    # Transpose HWC (224, 224, 3) -> CHW (3, 224, 224)
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension -> (1, 3, 224, 224)
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)
    return batched


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR numpy array.
    Falls back to pure NumPy channel-swap when cv2 is unavailable.
    """
    rgb_array = np.array(image)
    if _CV2_AVAILABLE and cv2 is not None:
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    # Pure NumPy fallback: reverse channel axis R,G,B -> B,G,R
    return rgb_array[:, :, ::-1].copy()


def cv2_to_pil(bgr_array: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR numpy array to PIL RGB image.
    Falls back to pure NumPy channel-swap when cv2 is unavailable.
    """
    if _CV2_AVAILABLE and cv2 is not None:
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    else:
        rgb_array = bgr_array[:, :, ::-1].copy()
    return Image.fromarray(rgb_array)


def compute_image_metrics(image: Image.Image) -> dict:
    """
    Extract basic computer vision metrics from the image
    (mean brightness, contrast/sharpness via Laplacian variance, aspect ratio).
    Works with or without cv2.
    """
    gray_np = _pil_to_gray_np(image)
    brightness = float(np.mean(gray_np))
    contrast = float(np.std(gray_np))
    width, height = image.size

    if _CV2_AVAILABLE and cv2 is not None:
        sharpness = float(cv2.Laplacian(gray_np, cv2.CV_64F).var())
    else:
        sharpness = _laplacian_variance(gray_np)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(height, 1), 2),
        "mean_brightness": round(brightness, 1),
        "sharpness_score": round(sharpness, 1),
        "contrast_score": round(contrast, 1)
    }


# ==============================================================================
# OCR PREPROCESSING & QUALITY CHECK PIPELINE
# ==============================================================================

def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
    """
    Calculate image quality indicators before OCR:
    - resolution, blur (Laplacian variance), brightness, contrast.
    Provides actionable suggestions if quality is suboptimal.
    Works with or without cv2.
    """
    metrics = compute_image_metrics(image)
    w, h = metrics["width"], metrics["height"]
    brightness = metrics["mean_brightness"]
    sharpness = metrics["sharpness_score"]
    contrast = metrics["contrast_score"]

    issues = []
    suggestions = []

    if w < 400 or h < 400:
        issues.append("Low Resolution")
        suggestions.append("Move the camera closer to capture fine label text in high detail.")

    if sharpness < 45.0:
        issues.append("Blurry Image")
        suggestions.append("Hold the camera steady or rest the package on a flat surface to reduce motion blur.")

    if brightness < 60.0:
        issues.append("Under-exposed / Dark")
        suggestions.append("Use better lighting or enable your room light/flashlight.")
    elif brightness > 220.0:
        issues.append("Over-exposed / Glare")
        suggestions.append("Reduce glare or tilt the package slightly away from direct reflections.")

    if contrast < 25.0:
        issues.append("Low Contrast")
        suggestions.append("Ensure the label text clearly stands out against the package background.")

    if not issues:
        quality_status = "Good"
        quality_desc = "Image quality is suitable for OCR extraction."
    elif len(issues) == 1:
        quality_status = f"Fair ({issues[0]})"
        quality_desc = f"Moderate quality issue detected: {issues[0]}."
    else:
        quality_status = f"Poor ({', '.join(issues)})"
        quality_desc = "Multiple quality issues detected. Preprocessing will attempt enhancement."

    if not suggestions:
        suggestions.append("Label is clear and ready for OCR analysis.")

    return {
        "quality_status": quality_status,
        "quality_desc": quality_desc,
        "is_acceptable": sharpness >= 20.0 and 30.0 <= brightness <= 245.0,
        "metrics": metrics,
        "issues": issues,
        "suggestions": suggestions
    }


def deskew_image(gray_np: np.ndarray) -> np.ndarray:
    """
    Estimate text skew angle and deskew if moderate.
    Requires cv2; returns input unchanged when cv2 is unavailable.
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return gray_np
    try:
        thresh = cv2.bitwise_not(cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return gray_np
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if 0.5 < abs(angle) < 20.0:
            (h, w) = gray_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(gray_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed
    except Exception:
        pass
    return gray_np


def preprocess_for_ocr(
    image: Image.Image,
    target_width: int = 1400
) -> Dict[str, Any]:
    """
    Comprehensive Image Preprocessing for Food Label OCR.
    Uses cv2 when available; falls back to pure PIL/NumPy on Streamlit Cloud
    or any headless environment where cv2 is not installed.

    Returns dict with keys:
        original, resized, grayscale, contrast_enhanced,
        sharpened, otsu_thresh, adaptive_thresh, deskewed,
        processed_primary, diagnostics
    """
    img_rgb = load_image(image)

    w, h = img_rgb.size
    if w != target_width and target_width > 0:
        aspect = h / max(w, 1)
        new_w = target_width
        new_h = int(target_width * aspect)
        img_resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        img_resized = img_rgb

    gray_np = _pil_to_gray_np(img_resized)

    if _CV2_AVAILABLE and cv2 is not None:
        # ---- Full cv2 pipeline ----
        denoised = cv2.bilateralFilter(gray_np, d=5, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
        _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        deskewed_gray = deskew_image(contrast_enhanced)
    else:
        # ---- Pure PIL / NumPy fallback ----
        denoised = _pil_bilateral_approx(gray_np)
        contrast_enhanced = _pil_clahe_approx(denoised)
        sharpened = _pil_sharpen(contrast_enhanced)
        otsu_thresh = _pil_otsu_threshold(sharpened)
        adaptive_thresh = _pil_adaptive_threshold(sharpened, block_size=21, c=10)
        deskewed_gray = contrast_enhanced  # deskew skipped without cv2

    quality_diagnostics = assess_image_quality(img_rgb)

    return {
        "original": img_rgb,
        "resized": img_resized,
        "grayscale": _gray_np_to_pil(gray_np),
        "contrast_enhanced": _gray_np_to_pil(contrast_enhanced),
        "sharpened": _gray_np_to_pil(sharpened),
        "otsu_thresh": _gray_np_to_pil(otsu_thresh),
        "adaptive_thresh": _gray_np_to_pil(adaptive_thresh),
        "deskewed": _gray_np_to_pil(deskewed_gray),
        "processed_primary": _gray_np_to_pil(contrast_enhanced),
        "diagnostics": quality_diagnostics
    }
