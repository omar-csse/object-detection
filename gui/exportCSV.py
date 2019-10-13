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

    def __init__(self, detectedStats):
        super().__init__()
        self.detectedStats = detectedStats

    def run(self):
        self.saveVideo()
        
    def saveCSV(self):
        for i, frame in enumerate(self.detectedFrames):
            # frame = cv2.cvtColor(np.float32(frame), cv2.COLOR_RGB2BGR)
            self.video_writer.write(np.uint8(frame))
        self.video_writer.release()
        cv2.destroyAllWindows()

        self.doneSignal.emit("CSV file is exported")