import os
import pathlib
import cv2
from PyQt5.QtCore import Qt, QThread


class Frame(QThread): 

    def __init__(self, videoPath, time):
        super().__init__()
        self.videoPath = videoPath
        self.videoTime = time
        self.video = cv2.VideoCapture(videoPath)
        self.videoName = os.path.basename(videoPath)
        self.framesPath = os.path.dirname(os.path.realpath(__file__)) + '/frames/' + self.videoName + '/'

    def run(self):

        if not os.path.exists(self.framesPath): 
            os.mkdir(self.framesPath)

        self.capture_frame()
        pass

    def capture_frame(self):
        self.video.set(cv2.CAP_PROP_POS_MSEC, self.videoTime)
        success, image = self.video.read()
        if success:
            print('frame has been captured')
            cv2.imwrite("{}/{}.jpg".format(self.framesPath, self.videoTime/1000), image)
            cv2.waitKey()
