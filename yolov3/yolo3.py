# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:06:48 2019

@author: Nathan
"""

import cv2
import json
import time
import os
from tqdm import tqdm
from keras.models import load_model
from .utils.utils import get_yolo_boxes
from .utils.bbox import draw_boxes
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

class YOLOv3(QThread):

    doneSignal = pyqtSignal(str)
    frameSignal = pyqtSignal(int, int)
    predictionSignal = pyqtSignal(list, QImage, list, int)

    def __init__(self, videoPath):
        super().__init__()
        self.input_path = videoPath
        self.output_lst = []
        self.count_lst =[]
        self.bbox_images = []
        self.config_path  = os.path.dirname(os.path.realpath(__file__)) + "/config.json"
        
        self.video_reader = cv2.VideoCapture(self.input_path)
        self.nb_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    def run(self):

        self.frameSignal.emit(self.frame_w, self.frame_h)
        self.yolo3_predict()

    def convert_CVmatToQpixmap(self, CVmat):
        # CVmat to Qimage
        CVmat = cv2.cvtColor(CVmat, cv2.COLOR_RGB2BGR)
        height, width, dim = CVmat.shape
        bytesPerLine = dim * width
        qimg = QImage(CVmat.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qimg

    def yolo3_predict(self):        
        
        with open(self.config_path) as config_buffer:    
                config = json.load(config_buffer)
        
        ###############################
        #   Load the model
        ###############################
        os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
        infer_model = load_model(os.path.dirname(os.path.realpath(__file__)) + config['train']['saved_weights_name'])
        
        for i in tqdm(range(self.nb_frames)):
            _, image = self.video_reader.read()
                    
            frame_stats, counts, result_image = self.frame_predict(config, infer_model, image)
            print("Frame analysed")
            
            ##############################
            # SEND IMAGES WHERE DESIRED HERE
            ##############################
            # ie. use result_image (w/ bounding boxes) or image (w/o bounding boxes)
            # ie. use self.count_lst to display class counts for current frame
            qimg = self.convert_CVmatToQpixmap(result_image)
            self.predictionSignal.emit(list(result_image), qimg, frame_stats, i)
            
            # Add new data to global
            self.bbox_images.append(result_image)
            self.output_lst.append(frame_stats)
            self.count_lst.append(counts)

        self.doneSignal.emit("YOLOv3 prediction is done")
        self.video_reader.release() 
        
        return self.bbox_images, self.output_lst, self.count_lst
        
    def frame_predict(self, config, infer_model, image):
        counts = []
 
        ###############################
        #   Set some parameter
        ###############################       
        net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
        obj_thresh, nms_thresh = 0.5, 0.15
        #0.45

        ###############################
        #   Predict bounding boxes 
        ###############################

        height, width, channels = image.shape

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        ### GET BOX DATA
        labels = config['model']['labels']
        labels = sorted(labels)
        stats = []
        self.shark_count = 0
        self.surfer_count = 0
        self.dolphin_count = 0
        for i, box in enumerate(boxes):
            if box.get_score() > 0:
                #print(labels[box.get_label()])
                #print(box.get_score())
                #print("Coordinates: xmin-" + str(box.xmin) + " ymin-" + str(box.ymin) + " xmax-" + str(box.xmax) + " ymax-" + str(box.ymax))
                
                # Convert coordinates
                xmin = box.xmin/width
                xmax = box.xmax/width
                ymin = box.ymin/height
                ymax = box.ymax/height
                
                # Increment counts
                if labels[box.get_label()] == "shark":
                    self.shark_count += 1
                elif labels[box.get_label()] == "surfer":
                    self.surfer_count += 1
                elif labels[box.get_label()] == "dolphin":
                    self.dolphin_count += 1
                else:
                    print("Unrecognized label!")
                
                # Concatenate string
                # image_str += str(labels[box.get_label()]) + " " +  str(box.get_score()) + " " \
                #             + str(xmin) + " " + str(xmax) + " " + str(ymin) + " " + str(ymax) + " "
                image_str = [str(labels[box.get_label()]),str(box.get_score()),str(xmin),str(xmax),str(ymin),str(ymax)]
                stats.append(image_str)
                #print(image_str)
                
        counts.append([self.shark_count, self.surfer_count, self.dolphin_count])
            
        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)   
                
        return stats, counts, image

    def save_video(self, images, input_path, output_path):
        # Save video to file when desired
        video_out = output_path + self.input_path.split('/')[-1]
        self.video_reader = cv2.VideoCapture(self.input_path)
        frame_h = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(video_out,
                                cv2.VideoWriter_fourcc(*'MPEG'), 
                                50.0, 
                                (frame_w, frame_h))
        
        for i in range(len(images)):
            # write result to the output video
            video_writer.write(images[i]) 
        video_writer.release()
        self.video_reader.release()