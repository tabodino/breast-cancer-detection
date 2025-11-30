"""Image preprocessing utilities."""

import numpy as np
import cv2


def preprocess_image(image: np.ndarray, target_size=(224, 224), enhance_contrast=True):
    """
    Preprocess image for model input.

    Args:
        image: Input image as numpy array
        target_size: Target size (height, width)
        enhance_contrast: Whether to apply histogram equalization

    Returns:
        Preprocessed image ready for model
    """
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Enhance contrast if requested
    if enhance_contrast:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Merge channels
        lab = cv2.merge([l_channel, a_channel, b_channel])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize to target size
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)

    return image_batch


def denormalize_image(image: np.ndarray):
    """
    Convert normalized image back to uint8 for display.

    Args:
        image: Normalized image

    Returns:
        Image as uint8
    """
    image = (image * 255).astype(np.uint8)
    return image
