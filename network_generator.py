import os 
from ctypes import *                                               # Import libraries
import math
import random
import cv2
import numpy as np
import time
import darknet
from collections import deque
from matplotlib import pyplot as plt

class DarknetNetwork():
    def __init__(self, configPath, weightPath, metaPath):
        netMain = None
        metaMain = None
        altNames = None                                # Path to meta data
        if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath)+"`")
        if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
            netMain = darknet.load_net_custom(configPath.encode( 
                "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
        if metaMain is None:
            metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        
        self.network, self.class_names, self.class_colors = darknet.load_network(configPath, metaPath, weightPath, batch_size=1) #(network, class_names, class_colors)


    def get_predictions(self,img,thresh=0.5):
        h,w = img.shape[:2]
        darknet_image = darknet.make_image(w, h, 3)
        darknet.copy_image_from_bytes(darknet_image, img.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=thresh)
        return detections #(label, confidence, bbox)
    

def convertToXYminmax(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, xmax, ymin, ymax

def drawBox(img, detections):
    for label, conf, bbox in detections:
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin, xmax, ymin, ymax = convertToXYminmax(x,y,w,h)
        p1 = (xmin,ymin)
        p2 = (xmax,ymax)
        color = (100,100,100)
        img = cv2.rectangle(img, p1, p2, color, 2)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)
        text_w, text_h = text_size

        cv2.rectangle(img, p1, (p1[0]+text_w+3, p1[1]-text_h-3), color, -1)
        cv2.putText(img, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                      (255,255,255), 2)
    return img

