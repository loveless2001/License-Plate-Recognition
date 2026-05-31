from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper
import function.ocr_variants as ocr_variants

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-o', '--output', help='path to save annotated output image')
args = ap.parse_args()

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_low_lr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

img = cv2.imread(args.image)
plates = yolo_LP_detect(img, size=640)

plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()
if len(list_plates) == 0:
    lp, tag = ocr_variants.read_plate_tta(yolo_license_plate, img)
    if lp != "unknown":
        label = ocr_variants.format_plate_tag(lp, tag)
        cv2.putText(img, label, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ocr_variants.tag_color(tag), 2)
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        lp, tag = ocr_variants.read_plate_tta(yolo_license_plate, crop_img)
        if lp != "unknown":
            list_read_plates.add(lp)
            label = ocr_variants.format_plate_tag(lp, tag)
            cv2.putText(img, label, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ocr_variants.tag_color(tag), 2)
if args.output:
    cv2.imwrite(args.output, img)
else:
    cv2.imshow('frame', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
