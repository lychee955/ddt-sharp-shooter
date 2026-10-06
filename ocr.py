import time
from skimage.feature import match_template
from typing import Tuple
import numpy as np
from logger import logger
import pyautogui
import ddddocr
import dotenv
import os
from numpy import ndarray
from skimage import color, measure, io
from PIL import Image

dotenv.load_dotenv()

_DEV = os.environ.get("DEV", "0") == "1"
_ocr = ddddocr.DdddOcr(show_ad=False)
_ocr.set_ranges("0123456789")
_throttle_key: int | None = None


def _recognize_digit(image: ndarray) -> str:
    # ndarray to PIL image
    image = np.array(image)
    image = Image.fromarray(image)
    result = _ocr.classification(image, probability=True)
    s = ""
    for i in result["probability"]:
        s += result["charsets"][i.index(max(i))]
    return s


def _capture_region(region: tuple[int, int, int, int], gray_scale=False) -> ndarray:
    global _throttle_key

    img_save_path = None
    if _DEV:
        img_save_path = f"{int(time.time())}.png"
        print(f"img_save_path: {img_save_path}")
        if _throttle_key is not None and _throttle_key == img_save_path:
            img_save_path = None
        _throttle_key = img_save_path
    # img_save_path = f"{time.time()}.png"

    img = pyautogui.screenshot(region=region, imageFilename=img_save_path)
    img = img.convert("RGB")
    if gray_scale:
        img = img.convert("L")
    img = np.array(img)
    return img


def _left_side_more_dark(image: ndarray) -> bool:
    image_bin = _binarize_image_by_reference(image, (141, 141, 232), 50)
    shape = image_bin.shape
    if _DEV:
        io.imsave("tmp_left_side_dark.png", image_bin)
    return bool(
        np.mean(image_bin[:, : shape[1] // 2]) > np.mean(image_bin[:, shape[1] // 2 :])
    )


def recognize(region: tuple[int, int, int, int] | np.ndarray) -> str:
    if isinstance(region, np.ndarray):
        image = region
    else:
        image = _capture_region(region)
    result = _recognize_digit(image)
    return result


def _recognize_red_wind(image: ndarray) -> float | None:
    """Read the red digits separately from the arrow and wind label."""
    pixels = image.astype(float)
    red = ((pixels[:, :, 0] > 150)
           & (pixels[:, :, 1] < pixels[:, :, 0] * .6)
           & (pixels[:, :, 2] < pixels[:, :, 0] * .6))
    components = measure.regionprops(measure.label(red))
    if not components:
        return None
    largest_height = max(p.bbox[2] - p.bbox[0] for p in components)
    digits = [p for p in components if p.bbox[2] - p.bbox[0] >= largest_height * .6 and p.area >= 4]
    if largest_height < 6 or len(digits) < 2:
        return None
    mask = np.zeros(red.shape, dtype=bool)
    for prop in digits:
        mask[tuple(prop.coords.T)] = True
    ys, xs = np.where(mask)
    binary = np.where(mask[ys.min():ys.max()+1, xs.min():xs.max()+1], 0, 255).astype(np.uint8)
    binary = np.pad(binary, 4, constant_values=255)
    text = _recognize_digit(binary)
    if not text.isdigit() or not 2 <= len(text) <= 3:
        raise ValueError("风力数字识别失败，请检查风力显示区域")
    return int(text) / 10


def recognize_wind(region: tuple[int, int, int, int] | ndarray) -> tuple[float, bool]:
    if isinstance(region, np.ndarray):
        image = region
    else:
        image = _capture_region(region)

    red_result = _recognize_red_wind(image)
    if red_result is not None:
        return red_result, _left_side_more_dark(image)
    image = image.copy()

    # before everything, try to remove all red color to improve the recognition
    image_bin = _binarize_image_by_reference(np.array(image), (200, 6, 15), 50)
    image[image_bin == 1] = [255, 255, 255]

    result = _recognize_digit(image)
    if not result.isdigit() or not 2 <= len(result) <= 3:
        raise ValueError("风力数字识别失败，请重新标记游戏区域，检查风力显示是否完整")
    if len(result) == 3:
        result = result[0] + result[2]

    try:
        result = int(result) / 10
    except ValueError as exc:
        raise ValueError("风力数字识别失败，请检查风力显示区域") from exc

    if result < 1:
        image_gray = color.rgb2gray(image)
        image_bin = image_gray > np.mean(image_gray) * 0.618
        image_bin = image_bin[:, : image_bin.shape[1] // 2]

        contours = measure.find_contours(image_bin, 0.8)
        if not contours:
            raise ValueError("风力整数位轮廓识别失败，请检查风力显示是否被遮挡")
        largest_contour = max(contours, key=len)
        min_x = int(np.min(largest_contour[:, 1]))
        max_x = int(np.max(largest_contour[:, 1]))
        w = max_x - min_x
        if w / image_bin.shape[1] < 0.37:
            result += 1

    if result % 1 == 0:
        image_gray = color.rgb2gray(image)
        image_bin = image_gray > np.mean(image_gray) * 0.618
        image_bin = image_bin[:, image_bin.shape[1] // 2 :]

        contours = measure.find_contours(image_bin, 0.8)
        if not contours:
            raise ValueError("风力小数位轮廓识别失败，请检查风力显示是否被遮挡")
        largest_contour = max(contours, key=len)
        min_x = int(np.min(largest_contour[:, 1]))
        max_x = int(np.max(largest_contour[:, 1]))
        w = max_x - min_x
        if w / image_bin.shape[1] < 0.37:
            result += 0.1
    return result, _left_side_more_dark(image)


def find_template(
    template: np.ndarray, region: Tuple[int, int, int, int] = None
) -> Tuple[int, int] | None:
    """
    Find the coordinates of the best match for a template image within a region.

    Args:
        template (ndarry): template image.
        region (Tuple[int, int, int, int], optional): Region to search (left, top, width, height).

    Returns:
        Tuple[int, int]: Coordinates of the best match's center.
    """

    # Capture the region of the screen
    screenshot_array = _capture_region(region, True)

    # Perform template matching
    result = match_template(screenshot_array, template)

    # The confidence of result should be better then 0.8
    if np.max(result) < 0.8:
        logger.warning("Template matches none result.")
        return None

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    # Calculate the center of the matched region
    center_x = x + template.shape[1] // 2
    center_y = y + template.shape[0] // 2

    return center_x, center_y


def _binarize_image_by_reference(
    image: ndarray, reference: tuple[int, int, int], threshold: float
) -> ndarray:
    """
    Binarize the image based on the reference color.

    Args:
        image (ndarray): The input image (should be rgb image).
        reference (tuple[int, int, int]): The reference color (R, G, B).
        threshold (float): The threshold for binarization.

    Returns:
        ndarray: The binarized image.
    """
    distances = np.sqrt(np.sum((image - reference) ** 2, axis=2))
    mask = distances <= threshold
    image = np.where(mask[:, :, np.newaxis], 0, 1)
    image = image[:, :, 0]
    image = np.logical_not(image)
    return image


def _recognize_rect_width(image: ndarray) -> int:
    """
    Recognize width of the rectangle in the image.

    Args:
        image (ndarray): The input image (should be rgb image).

    Returns:
        int: The width of the rectangle, or 0 if no valid width is found.
    """
    target_color = np.array([160, 163, 169])
    image = _binarize_image_by_reference(image, target_color, 40)

    if _DEV:
        io.imsave("tmp.png", image)

    # The full minimap also has a grey outer frame. Identify a horizontal
    # viewport edge with two vertical sides instead of using the outermost
    # grey columns (which would measure the entire map).
    height, width = image.shape
    minimum_height = max(8, height // 4)
    best_score, left_side, right_side = 0, 0, 0
    for y, row in enumerate(image):
        edges = np.diff(np.r_[False, row, False].astype(int))
        for left, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            right = stop - 1
            if not max(12, width * .1) <= right-left < width * .9:
                continue
            # Antialiasing/background can extend a horizontal run a few
            # pixels past its vertical border. Align each end to an actual
            # vertical line, rather than accepting an unrelated grey patch.
            margin = max(2, round(height/60))
            offsets = [0] + [offset for d in range(1, margin+1) for offset in (-d, d)]
            columns = [np.array([x+d for d in offsets if 0 <= x+d < width])
                       for x in (left, right)]
            for direction in (-1, 1):
                available = y if direction == -1 else height-y-1
                if available < minimum_height:
                    continue
                indices = y + direction * np.arange(1, available+1)
                lengths = np.arange(1, available+1)
                side_coverage = [np.cumsum(image[indices[:, None], cols], axis=0)/lengths[:, None]
                                 for cols in columns]
                coverage = np.minimum(*(values.max(axis=1) for values in side_coverage))
                valid = (lengths >= minimum_height) & (coverage >= .7)
                score = float(np.max(np.where(valid, coverage*lengths, 0)))
                if score > best_score:
                    best = int(np.argmax(np.where(valid, coverage*lengths, 0)))
                    left_side, right_side = (int(cols[np.argmax(values[best])])
                                            for cols, values in zip(columns, side_coverage))
                    best_score = score
    w = right_side - left_side

    if _DEV:
        # draw a rectangle on the image with matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        print(f"left_side: {left_side}, right_side: {right_side}, w: {w}")

        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        rect = patches.Rectangle(
            (left_side, 0),
            w,
            image.shape[1] - 1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.show()

    return w


def recognize_ten_units(region: tuple[int, int, int, int] | ndarray) -> int:
    """
    Recognize the ten units in the image.

    Args:
        region (tuple[int, int, int, int]): The region to capture.

    Returns:
        int: The pixel width corresponding to ten units.
    """
    if isinstance(region, np.ndarray):
        image = region
    else:
        image = _capture_region(region)
    return _recognize_rect_width(image)


def recognize_force(region: tuple[int, int, int, int] | ndarray) -> int:
    """
    Recognize the force in the image.

    Args:
        region (tuple[int, int, int, int] | ndarray): The region to capture (or directly the image) and recognize.

    Returns:
        int: The force.
    """
    if isinstance(region, np.ndarray):
        image = region
    else:
        image = _capture_region(region)

    image_bin = _binarize_image_by_reference(image, (232, 180, 70), 61.8)

    if _DEV:
        io.imsave("tmp_force.png", image_bin)

    right_side = 0
    for i in range(image_bin.shape[1] - 1, -1, -1):
        if np.sum(image_bin[:, i]) > image_bin.shape[0] / 3:
            right_side = i
            break

    return right_side / image_bin.shape[1] * 100


if __name__ == "__main__":
    # image = io.imread("wind.png")
    # res = recognize_wind(image)
    # print(res)

    # image = io.imread("minimap.png")
    # res = recognize_ten_units(image)
    # print(res)

    # image = io.imread("deg.png")
    # res = recognize(image)
    # print(res)

    # image = io.imread("force_1.png")
    # res = recognize_force(image)
    # print(res)
    pass
