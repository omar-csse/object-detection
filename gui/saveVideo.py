import os
import pathlib
import time
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

class SaveVideo(QThread): 

    doneSignal = pyqtSignal(str)

    def __init__(self, videoPath, frames, frame_w, frame_h):
        super().__init__()
        self.detectedFrames = frames
        print(videoPath)
        self.video_writer = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc(*'XVID'), 24, (frame_w, frame_h))

    def run(self):
        self.saveVideo()
        
    def saveVideo(self):
        for i, frame in enumerate(self.detectedFrames):
            frame = cv2.cvtColor(np.float32(frame), cv2.COLOR_RGB2BGR)
            self.video_writer.write(np.uint8(frame))
        self.video_writer.release()
        cv2.destroyAllWindows()

        self.doneSignal.emit("video is saved")