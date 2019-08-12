import os
import cv2
import json
import sys
import random
from os import listdir


class Frames():

    """Frames model to capture specified number of frames in videos."""

    def __init__(self, start_number=1):
        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self._log_file = None
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

    def _cleanDirectories(self, clean_logs=False):
        if os.path.exists(self._dir_path + '/logs/log.txt'):
            os.remove(self._dir_path + '/logs/log.txt')

    def openLogs(self):
        self._cleanDirectories()
        if not os.path.exists(self._dir_path+'/logs'): 
            os.makedirs(self._dir_path+'/logs')
        self._log_file = open(self._dir_path + '/logs/log.txt', 'a')
        return self._log_file

    def closeLogs(self):
        self._log_file.close()

    def setup(self, objectType='surfers'):

        if not os.path.exists(self._dir_path+'/frames'): 
            os.mkdir(self._dir_path+'/frames')
        
        self._log_file.write("""Frames model to capture specified number of frames in videos.""")
        self._log_file.write("\n\n\n")
        if not os.path.exists(self._dir_path+'/frames/' + objectType): 
            os.mkdir(self._dir_path+'/frames/'+objectType)

        self._frames_folder = self._dir_path + '/frames/' + objectType
        self.object_dir = self._dir_path + '/videos/' + objectType
        self.objectVideos = [self.object_dir+'/'+f for f in listdir(self.object_dir)]

    def capture_frames(self, cap_per_frame=5):

        for video in self.objectVideos:

            if not os.path.exists(self._frames_folder + '/' + os.path.basename(video)): 
                os.mkdir(self._frames_folder + '/' + os.path.basename(video))
            
            video_reader = cv2.VideoCapture(video)
            total_caps = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_reader.get(cv2.CAP_PROP_FPS))

            if cap_per_frame is 'min': cap_per_frame = fps
            elif cap_per_frame is 'max': cap_per_frame = 1

            cap_count = 0
            print("\n{}\n\n".format(os.path.basename(video)))
            self._log_file.write("\nVideo name: {}\n\n".format(os.path.basename(video)))
            print('Number of frames to process: {}'.format(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)))
            self._log_file.write('Number of frames to process: {}\n'.format(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)))
            print('Number of fps to capture: {}\n\n'.format(fps))
            self._log_file.write('Number of fps to capture: {}\n\n'.format(fps))
            
            for i in range(total_caps):
                ret, image = video_reader.read()
                if i % cap_per_frame == 0:
                    cv2.imwrite('{}/{}/{}.png'.format(self._frames_folder, os.path.basename(video), cap_count), image)
                    print('frame#:{} - captur: #{} for {}'.format(i, cap_count, os.path.basename(video)))
                    self._log_file.write('frame#:{} - captur: #{} for {}\n'.format(i, cap_count, os.path.basename(video)))
                    cap_count = cap_count + 1
                        
            video_reader.release()
            self._log_file.write("\n\n")
            print("\n\n")


def main():
    frames = Frames(start_number=0)
    logs_file = frames.openLogs()

    frames.setup(objectType='surfers')
    frames.capture_frames(cap_per_frame='min')

    frames.setup(objectType='sharks')
    frames.capture_frames(cap_per_frame=5)

    logs_file.write('\n\nCapturing frames in all videos have been done')
    frames.closeLogs()

if __name__ == "__main__":
    main()