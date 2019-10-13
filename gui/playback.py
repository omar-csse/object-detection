import os
import pathlib
import time
import cv2
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

class PlayBack(QThread): 

    Stop = False
    Pause = False
    doneSignal = pyqtSignal(str)
    imageSignal = pyqtSignal(list, QImage, list, int, bool)
    frameSignal = pyqtSignal(int, int)
    errorSignal = pyqtSignal(str)

    def __init__(self, videoPath, csvPath, detectedStats, detectedFrames):
        super().__init__()
        self.csvPath = csvPath
        self.detectedStats = detectedStats
        self.detectedFrames = detectedFrames
        self.videoPath = videoPath
        self.threadactive = True
        self.video_reader = cv2.VideoCapture(r'{}'.format(videoPath))
        self.nb_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.classes = []
        self.statisticsInFrme = []

    def run(self):
        self.frameSignal.emit(self.frame_w, self.frame_h)

        if self.detectedStats: 
            self.csv = False
            self.start_playback(self.csv, len(self.detectedFrames))
        elif self.csvPath:
            self.csv = True
            self.readCSV()
            self.start_playback(self.csv, self.nb_frames)

    def readCSV(self):
        self.labels = pd.read_csv(self.csvPath)
        self.expLabels = self.labels['PredictionString'].str.split(' ', expand=True)

        for i in range(0, self.expLabels.shape[1], 6):
            self.classes.append(self.expLabels[i].unique().tolist())

        self.classes = set(x for l in self.classes for x in l)
        self.classes = list(filter(None, self.classes))
        self.colours = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def drawCsvAnnotations(self, data, expandedData, frameNumber, frame):
        self.statisticsInFrme = []
        item = data[data.FrameNumber == frameNumber]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for index, row in item.iterrows():
            for i in range(0, expandedData.shape[1], 6):
                if expandedData[i][index] == None or not expandedData[i][index]:
                    break # it breaks the loop if there are no more objects to draw
                colour = self.colours[self.classes.index(expandedData[i][index])]
                objClass = expandedData[i][index]
                confidence = float(expandedData[i + 1][index])
                xMin = int(float(expandedData[i + 2][index]) * frame.shape[1])
                xMax = int(float(expandedData[i + 3][index]) * frame.shape[1])
                yMin = int(float(expandedData[i + 4][index]) * frame.shape[0])
                yMax = int(float(expandedData[i + 5][index]) * frame.shape[0])
                cv2.rectangle(frame,(xMin, yMin),(xMax, yMax),colour,3)
                st_row = [objClass, str(confidence), str(xMin), str(xMax), str(yMin), str(yMax)]
                self.statisticsInFrme.append(st_row)
                y = yMin - 25 if yMin - 25 > 25 else yMin + 25
                label = "{}: {:.2f}%".format(objClass, confidence * 100)
                cv2.putText(frame, label, (xMin + 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        
        return frame

    def convert_CVmatToQpixmap(self, CVmat):
        # CVmat to Qimage
        height, width, dim = CVmat.shape
        bytesPerLine = dim * width
        qimg = QImage(CVmat.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qimg

    def start_playback(self, csv, nb_frames):
        try:
            print("Playback started...")
            print("Total number of frames: {}".format(nb_frames))
            self.i = 0
            while (self.i < nb_frames and self.Stop is False):
                while (self.Pause): pass
                if csv is True:
                    ret, frame = self.video_reader.read()
                    image = self.drawCsvAnnotations(self.labels, self.expLabels, self.i, frame)
                else:
                    image = np.array(self.detectedFrames[self.i])
                qimage = self.convert_CVmatToQpixmap(image)
                time.sleep(1/35)
                if self.Stop == False:
                    if csv is True: self.imageSignal.emit(list(image), qimage, self.statisticsInFrme, self.i, True)
                    else: self.imageSignal.emit(list(self.detectedFrames[self.i]), qimage, self.detectedStats[self.i], self.i, False)
                self.i = self.i + 1
            self.i = 0
            self.doneSignal.emit("Video finished playingback")
            print("Playback done...")
            # release resources
            self.video_reader.release()
            cv2.destroyAllWindows()
        except Exception as e: 
            self.errorSignal.emit("Invalid CSV data")

    def stop(self):
        self.doneSignal.emit("Video finished playingback")
        self.Stop = True
        self.Pause = False
        self.wait()

    def single_frame(self, csv, i):
        if csv is True:
            ret, frame = self.video_reader.read()
            image = self.drawCsvAnnotations(self.labels, self.expLabels, self.i, frame)
        else:
            image = np.array(self.detectedFrames[self.i])
        qimage = self.convert_CVmatToQpixmap(image)
        if self.Stop == False:
            if csv is True: 
                self.imageSignal.emit(list(image), qimage, self.statisticsInFrme, self.i, True)
            else: self.imageSignal.emit(list(self.detectedFrames[self.i]), qimage, self.detectedStats[self.i], self.i, False)

    def nextFrame(self):
        self.i = self.i + 1
        if self.i > self.nb_frames: self.i = self.nb_frames - 1
        self.Pause = True
        self.single_frame(self.csv, self.i)

    def prevFrame(self):
        self.i = self.i - 1
        if self.i < 0: self.i = 0
        self.Pause = True
        self.single_frame(self.csv, self.i)