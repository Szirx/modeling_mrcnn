import cv2
import numpy as np
from PIL import Image


def draw_random_masks(
    image: Image,
    masks: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.5,
    mask_threshold: float = 0.1,
):
    post_process_masks = []
    for score, mask in zip(scores, masks):
        if score < min_score:
            continue
        mask = mask > mask_threshold
        post_process_masks.append(mask)
    if len(post_process_masks) > 0:
        post_process_masks = np.squeeze(np.asarray(post_process_masks))

    alpha = 0.3
    image = np.asarray(image)
    masked_image = image.copy()
    max_value = 255

    if len(post_process_masks.shape) == 2:
        post_process_masks = np.expand_dims(post_process_masks, axis=0)
    for mask in post_process_masks:
        masked_image = np.where(
            np.repeat(mask[:, :, np.newaxis], 3, axis=2),
            np.random.randint(0, max_value, size=(3)),
            masked_image,
        )
        masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(image, alpha, masked_image, 1 - alpha, 0, dtype=cv2.CV_8U)