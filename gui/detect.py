import os
import pathlib
import cv2
from PyQt5.QtCore import Qt, QThread
import numpy as np

class Detect(QThread): 

    def __init__(self, videoPath):
        super().__init__()
        self.videoPath = videoPath
        self.video_out = os.path.dirname(os.path.realpath(__file__)) + '/test_data/predicted.mp4'
        self.video_reader = cv2.VideoCapture(videoPath)
        self.videoName = os.path.basename(videoPath)
        self.yolov3_cfg = os.path.dirname(os.path.realpath(__file__)) + '/../../example/yolov3.cfg'
        self.yolov3_classes = os.path.dirname(os.path.realpath(__file__)) + '/../../example/yolov3.txt'
        self.classes = None
        self.index = 0
        self.yolov3_weights = os.path.dirname(os.path.realpath(__file__)) + '/../../example/yolov3.weights'

    def run(self):

        with open(self.yolov3_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.scale = 0.00392
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

        # generate different colors for different classes 
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.nb_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        self.frame_h = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.video_writer = cv2.VideoWriter(self.video_out, cv2.VideoWriter_fourcc(*'XVID'), 24, (self.frame_w, self.frame_h))

        self.start_detection()
        pass


    def get_output_layers(self, net): 
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.colors[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def processImage(self, image, index):

        Width = image.shape[1]
        Height = image.shape[0]

        # read pre-trained model and config file
        net = cv2.dnn.readNet(self.yolov3_weights, self.yolov3_cfg)

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)
        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(self.get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
        
            self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
        return image

    def start_detection(self):
        self.index = 0
        print("Detection started...")
        print("Total number of frames: {}".format(self.nb_frames))

        for i in range(self.nb_frames):

            print("detection - current frame: {}".format(i))

            ret, image = self.video_reader.read()
            input_image = cv2.resize(image, (416, 416))
            input_image = input_image / 255.
            input_image = input_image[:,:,::-1]
            input_image = np.expand_dims(input_image, 0)

            image = self.processImage(image, self.index)
            self.video_writer.write(np.uint8(image))
            self.index = self.index +1

        print("Detection done...")
        # release resources
        self.video_reader.release()
        self.video_writer.release()
        cv2.destroyAllWindows()