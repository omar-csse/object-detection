from __future__ import division
import os
import glob
import cv2
import numpy as np

import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
print(sys.path)

import pickle
import time
from optparse import OptionParser
from .keras_frcnn import config
from .keras_frcnn import roi_helpers
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from .init import predict
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage
 
class InceptionV4(QThread):

    doneSignal = pyqtSignal(str, bool)
    frameSignal = pyqtSignal(int, int)
    predictionSignal = pyqtSignal(list, QImage, list, int, bool)

    def __init__(self, videoPath):
        super().__init__()
        self.input_path = videoPath
        self.video_reader = cv2.VideoCapture(self.input_path)
        self.nb_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    def run(self):

        self.frameSignal.emit(self.frame_w, self.frame_h)
        self.frcnn_predict()


    def frcnn_img (self, imgpath):
    #     imgin = cv2.imread(imgpath)
    #     (imgout, bbox) = predict(1,imgin)
    #     return imgout,bbox
        pass

    def convert_CVmatToQpixmap(self, CVmat):
        # CVmat to Qimage
        height, width, dim = CVmat.shape
        bytesPerLine = dim * width
        qimg = QImage(CVmat.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qimg

    def frcnn_predict (self):

        frames_out = []
        b_box = []
        print('number of frames:', self.nb_frames)

        for i in range(self.nb_frames):
            ret, image = self.video_reader.read()
            (frame,bbox_temp) = predict(i, image)
            frames_out.append(frame)
            b_box.append(bbox_temp)
            self.predictionSignal.emit(list(frames_out), self.convert_CVmatToQpixmap(frames_out), b_box, i, False)
        self.doneSignal.emit("InceptionV4 prediction is done", True)
        K.clear_session()
        return (frames_out,b_box)

        self.video_reader.release()