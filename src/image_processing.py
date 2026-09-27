"""
Image preprocessing and computer vision transformation utilities.
Handles image loading, color-space conversion, resizing, and normalization
for deep learning inference and optical character recognition (OCR) pipelines.
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
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    rgb_array = np.array(image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(bgr_array: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL RGB image."""
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def compute_image_metrics(image: Image.Image) -> dict:
    """
    Extract basic computer vision metrics from the image
    (mean brightness, contrast/sharpness via Laplacian variance, aspect ratio).
    """
    img_bgr = pil_to_cv2(image)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    # Laplacian variance gives a measure of focus / sharpness
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    width, height = image.size

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
    - resolution
    - blur (Laplacian variance)
    - brightness (mean grayscale intensity)
    - contrast (standard deviation of intensity)

    Provides actionable suggestions if quality is suboptimal.
    """
    metrics = compute_image_metrics(image)
    w, h = metrics["width"], metrics["height"]
    brightness = metrics["mean_brightness"]
    sharpness = metrics["sharpness_score"]
    contrast = metrics["contrast_score"]

    issues = []
    suggestions = []

    # Resolution check
    if w < 400 or h < 400:
        issues.append("Low Resolution")
        suggestions.append("Move the camera closer to capture fine label text in high detail.")

    # Blur check (Laplacian variance < 50 typically indicates noticeable blur)
    if sharpness < 45.0:
        issues.append("Blurry Image")
        suggestions.append("Hold the camera steady or rest the package on a flat surface to reduce motion blur.")

    # Lighting / Brightness check
    if brightness < 60.0:
        issues.append("Under-exposed / Dark")
        suggestions.append("Use better lighting or enable your room light/flashlight.")
    elif brightness > 220.0:
        issues.append("Over-exposed / Glare")
        suggestions.append("Reduce glare or tilt the package slightly away from direct reflections.")

    # Contrast check
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


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """
    Estimate text skew angle via minimum area bounding box of foreground pixels
    and deskew if angle is moderate.
    """
    try:
        # Invert for white foreground on black background
        thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if slight tilt (between 0.5 and 20 degrees)
        if 0.5 < abs(angle) < 20.0:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed
    except Exception:
        pass
    return gray


def preprocess_for_ocr(
    image: Image.Image,
    target_width: int = 1400
) -> Dict[str, Any]:
    """
    Comprehensive Computer Vision Preprocessing for Food Label OCR:
    1. RGB conversion & EXIF correction.
    2. Proportional resizing to optimal OCR resolution (~1400px width).
    3. Grayscale conversion.
    4. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    5. Sharpening filter to delineate letter contours.
    6. Non-local means / bilateral noise reduction.
    7. Otsu's Global Thresholding (Binarization).
    8. Gaussian Adaptive Thresholding.
    9. Optional deskewing.

    Returns dictionary containing both original and processed stages
    so the data science pipeline can be visually inspected by the user.
    """
    # 1. Base RGB Image
    img_rgb = load_image(image)

    # 2. Resize while maintaining aspect ratio if image is too small or overly large
    w, h = img_rgb.size
    if w != target_width and target_width > 0:
        aspect = h / max(w, 1)
        new_w = target_width
        new_h = int(target_width * aspect)
        img_resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        img_resized = img_rgb

    # Convert to OpenCV BGR and Grayscale
    bgr = pil_to_cv2(img_resized)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 3. Noise reduction (slight Gaussian/Bilateral filtering to remove print grain)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. Contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # 5. Sharpening kernel
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)

    # 6. Global Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 7. Adaptive Gaussian Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

    # 8. Deskewing
    deskewed_gray = deskew_image(contrast_enhanced)

    # 9. Quality Assessment Diagnostics
    quality_diagnostics = assess_image_quality(img_rgb)

    # Convert OpenCV matrices back to PIL Images for display
    return {
        "original": img_rgb,
        "resized": img_resized,
        "grayscale": Image.fromarray(gray),
        "contrast_enhanced": Image.fromarray(contrast_enhanced),
        "sharpened": Image.fromarray(sharpened),
        "otsu_thresh": Image.fromarray(otsu_thresh),
        "adaptive_thresh": Image.fromarray(adaptive_thresh),
        "deskewed": Image.fromarray(deskewed_gray),
        # Best processed image recommendation for OCR: contrast_enhanced / sharpened
        "processed_primary": Image.fromarray(contrast_enhanced),
        "diagnostics": quality_diagnostics
    }
