import cv2

import function.helper as helper
import function.utils_rotate as utils_rotate


NORMAL_TAG = "NORMAL"
GOV_TAG = "GOV"
MIL_TAG = "MIL"
OOD_INVERT_TAG = "OOD-INVERT"
OOD_OTSU_TAG = "OOD-OTSU"
OOD_CLAHE_INVERT_TAG = "OOD-CLAHE-INVERT"


def invert_gray(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)


def otsu_black_on_white(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2BGR)


def clahe_invert(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.cvtColor(255 - clahe, cv2.COLOR_GRAY2BGR)


def upscale(crop, scale):
    if scale == 1:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def has_min_chars(plate, min_chars):
    return len(plate.replace("-", "")) >= min_chars


def color_plate_tag(crop):
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    area = crop.shape[0] * crop.shape[1]

    red_mask = ((hue <= 12) | (hue >= 165)) & (saturation > 50) & (value > 50)
    blue_mask = (hue >= 85) & (hue <= 135) & (saturation > 40) & (value > 40)
    red_ratio = red_mask.sum() / area
    blue_ratio = blue_mask.sum() / area

    if red_ratio >= 0.35 and red_ratio > blue_ratio * 1.5:
        return MIL_TAG
    if blue_ratio >= 0.55 and blue_ratio > red_ratio * 1.5:
        return GOV_TAG
    return None


def semantic_plate_tag(crop, fallback_tag):
    return color_plate_tag(crop) or fallback_tag


def read_plate_deskewed(yolo_license_plate, crop, min_chars=6):
    for cc in range(0, 2):
        for ct in range(0, 2):
            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop, cc, ct))
            if lp != "unknown" and has_min_chars(lp, min_chars):
                return lp
    return "unknown"


def read_plate_tta(yolo_license_plate, crop):
    variants = [
        (NORMAL_TAG, crop, 6),
        (OOD_INVERT_TAG, invert_gray(crop), 6),
        (OOD_OTSU_TAG, otsu_black_on_white(crop), 6),
        (OOD_CLAHE_INVERT_TAG, clahe_invert(crop), 8),
        (OOD_CLAHE_INVERT_TAG, clahe_invert(upscale(crop, 2)), 8),
        (OOD_CLAHE_INVERT_TAG, clahe_invert(upscale(crop, 4)), 8),
    ]
    for tag, variant, min_chars in variants:
        lp = read_plate_deskewed(yolo_license_plate, variant, min_chars=min_chars)
        if lp != "unknown":
            return lp, semantic_plate_tag(crop, tag)
    return "unknown", None


def format_plate_tag(plate, tag):
    if tag == NORMAL_TAG:
        return plate
    return f"{tag}: {plate}"


def tag_color(tag):
    if tag == NORMAL_TAG:
        return (36, 255, 12)
    if tag == GOV_TAG:
        return (255, 128, 0)
    if tag == MIL_TAG:
        return (0, 0, 255)
    return (0, 215, 255)
