import os
import cv2
import json
import shutil
import sys
import random
from os import listdir
import xml.etree.ElementTree as ET


class Data():

    """Data class."""

    def __init__(self):
        self._dir_path = os.path.dirname(os.path.realpath(__file__))+ '/data'
        self._frames_path = os.path.dirname(os.path.realpath(__file__))+ '/frames'
        self._labels_path = os.path.dirname(os.path.realpath(__file__))+ '/labels'


    def setup(self, objectType):

        if not os.path.exists(self._dir_path+'/'+objectType): 
            os.makedirs(self._dir_path+'/'+objectType+'/frames')
            os.makedirs(self._dir_path+'/'+objectType+'/labels')

        self._frames = [self._frames_path+'/'+objectType+'/'+f for f in listdir(self._frames_path+'/'+objectType)]
        self._labels = [self._labels_path+'/'+objectType+'/'+f for f in listdir(self._frames_path+'/'+objectType)]
        self._objectdir = self._dir_path + '/' + objectType

    def rename_frames(self):
        # self.rename(self._frames)
        pass

    def rename_labels(self):
        self.rename(self._labels)
        self.edit_xml(self._labels)
        pass

    def rename(self, data):
        with open("logs.txt", "w") as text_file:
            for subdata_path in data:
                if "DS_Store" not in subdata_path:
                    for frame in os.listdir(subdata_path): 
                        old_name = subdata_path + "/" + frame
                        new_name = subdata_path + "/" + os.path.basename(subdata_path) + "_" + frame
                        text_file.write("old: " + old_name + "\nnew: " + new_name + "\n\n\n")
                        # os.rename(old_name, new_name)
    
    def edit_xml(self, data):
        for subdata_path in data:
                if "DS_Store" not in subdata_path:
                    for label in os.listdir(subdata_path): 
                        if "DS_Store" not in subdata_path + "/" + label:
                            tree = ET.parse(subdata_path + "/" + label)
                            root = tree.getroot()
                            newfilename = os.path.basename(subdata_path) + "_" + label
                            newpath = subdata_path + "/" + os.path.basename(subdata_path) + "_" + label
                            root[1].text = newfilename
                            root[2].text = newpath
                            tree.write(self._objectdir+'/labels/'+newfilename)


def main():
    data = Data()
    data.setup(objectType='sharks')
    # data.rename_frames()
    # data.rename_labels()

if __name__ == "__main__":
    main()
