import os
import pathlib
import time
import csv
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

class ExportCSV(QThread): 

    doneSignal = pyqtSignal(str)
    errorSignal = pyqtSignal(str)

    def __init__(self, videoName, detectedStats, frame_w, frame_h):
        super().__init__()
        self.importedVideo = videoName
        self.detectedStats = detectedStats
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.csvrows = []
        self.csvPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'out', '{}_predicted.csv'.format(self.importedVideo))

    def run(self):
        self.saveCSV()

    def saveCSV(self):
        try:
            for i, row in enumerate(self.detectedStats):
                    self.img_string = ""
                    if row:
                        for n, fram_stat in enumerate(row):
                            conf = "{0:.2f} ".format(float(fram_stat[1]))
                            xMin = "{0:.2f} ".format(float(int(fram_stat[2]) / self.frame_w))
                            xMax = "{0:.2f} ".format(float(int(fram_stat[3]) / self.frame_w))
                            yMin = "{0:.2f} ".format(float(int(fram_stat[4]) / self.frame_h))
                            yMax = "{0:.2f} ".format(float(int(fram_stat[5]) / self.frame_h))
                            self.img_string += str(fram_stat[0]) + " " + conf + xMin + xMax + yMin + yMax

                    self.csvrows.append([i, self.img_string])

            with open(self.csvPath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['FrameNumber', 'PredictionString'])
                    for i, csvrow in enumerate(self.csvrows):
                        writer.writerow(csvrow)

            self.doneSignal.emit("CSV file is exported")
        except Exception as e: 
            self.errorSignal.emit("Error while exporting CSV")