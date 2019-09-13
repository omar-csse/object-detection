import os
import cv2
import json
import shutil
import sys
import random
from os import listdir
import xml.etree.ElementTree as ET


class Group():

    """Data class."""

    def __init__(self):
        self._dir_path = os.path.dirname(os.path.realpath(__file__))+ '/data'

    def setup(self, objectType):
        self._frames_path = self._dir_path + '/' + objectType + '/frames'
        self._labels_path = self._dir_path + '/' + objectType + '/labels'
        self._frames = [f for f in listdir(self._frames_path)]
        self._labels = [f for f in listdir(self._labels_path)]

    def move_imgs(self):
        counter = 0
        for i,frame in enumerate(listdir(self._frames_path)):
            for j,label in enumerate(listdir(self._labels_path)):
                if (os.path.splitext(frame)[0] == os.path.splitext(label)[0]):
                    counter += 1
                    shutil.move(self._frames_path+'/'+frame, self._labels_path)
                    print("counter: " + str(counter) + " - " + frame)

def main():
    data = Group()
    data.setup(objectType='sharks')
    data.move_imgs()

if __name__ == "__main__":
    main()
