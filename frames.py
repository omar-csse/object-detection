import os
import cv2
import json
import sys
import random
from os import listdir


class Frames():

    def __init__(self, start_number=1, cap_pernumber_frame=5):

        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self.start_number = start_number
        self.cap_pernumber_frame = cap_pernumber_frame

    def setup(self, objectType='surfers'):

        if not os.path.exists(self._dir_path+'/frames'): 
            os.mkdir(self._dir_path+'/frames')

        self._frames_folder = self._dir_path + '/frames/' + objectType
        self.object_dir = self._dir_path + '/videos/' + objectType
        self.objectVideos = [self.object_dir+'/'+f for f in listdir(self.object_dir)]

    def capture_frames(self):

        for video in self.objectVideos:

            print("\n\n{}".format(os.path.basename(video)))
            print(self._frames_folder)
            
            video_reader = cv2.VideoCapture(video)

            fps = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = sum(x * int(t) for x, t in zip([3600, 60, 1], start_time.split(":"))) * 25
            end_frame = sum(x * int(t) for x, t in zip([3600, 60, 1], end_time.split(":")))  * 25
            nb_caps = (end_frame - start_frame) # cap_pernumber_frame
            count = 0
            cap_count = start_number
            print("\n\n{}".format(os.path.basename(video)))
            print('Number of frames to process: {}'.format(fps))
            print('Number of frames to capture: {}\n\n'.format(nb_caps))

            # for i in range(fps):
            #     ret, image = video_reader.read()
            #     if i > start_frame:
            #         count = count+1
            #         if count % self.cap_pernumber_frame == 0:
            #             out_image = cv2.resize(image, (3840, 2160))
            #             cv2.imwrite('{}/{}/{}.png'.format(self._frames_folder, os.path.basename(video), cap_count), out_image)
            #             #replace test{} by Dolphin{} etc.caps by dir you want
            #             print('Saving capture #{} for {}'.format(cap_count, os.path.basename(video)))
            #             cap_count = cap_count+1

            #     if i == end_frame:
            #         print('job done')
            #         break
                        
            # video_reader.release()


def main():
    frames = Frames(start_number=1, cap_pernumber_frame=5)

    frames.setup(objectType='surfers')
    frames.capture_frames()

    frames.setup(objectType='sharks')
    frames.capture_frames()

if __name__ == "__main__":
    main()