import cv2

import function.helper as helper
import function.utils_rotate as utils_rotate


NORMAL_TAG = "NORMAL"
OOD_INVERT_TAG = "OOD-INVERT"
OOD_OTSU_TAG = "OOD-OTSU"


def invert_gray(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)


def otsu_black_on_white(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2BGR)


def read_plate_deskewed(yolo_license_plate, crop):
    for cc in range(0, 2):
        for ct in range(0, 2):
            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop, cc, ct))
            if lp != "unknown":
                return lp
    return "unknown"


def read_plate_tta(yolo_license_plate, crop):
    variants = [
        (NORMAL_TAG, crop),
        (OOD_INVERT_TAG, invert_gray(crop)),
        (OOD_OTSU_TAG, otsu_black_on_white(crop)),
    ]
    for tag, variant in variants:
        lp = read_plate_deskewed(yolo_license_plate, variant)
        if lp != "unknown":
            return lp, tag
    return "unknown", None


def format_plate_tag(plate, tag):
    if tag == NORMAL_TAG:
        return plate
    return f"{tag}: {plate}"


def tag_color(tag):
    if tag == NORMAL_TAG:
        return (36, 255, 12)
    return (0, 215, 255)
