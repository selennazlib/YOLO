# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

path = 'C:\\Users\\S BASA\\Desktop\\YOLO\\yolov3-pretrained-image\\images\\cats.jpg'

img = cv2.imread(path)
# print(img) to crosscheck to see if we read the image right

img_width = img.shape[1]
img_height = img.shape[0]

img_blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
# swapRB=True -> from BGR to RGB
"""
Creates 4-dimensional blob from image.
Optionally resizes and crops image from center, 
subtract mean values, 
scales values by scalefactor, 
swap Blue and Red channels.

Parameters
image->input image (with 1-, 3- or 4-channels).

size->spatial size for output image

mean->scalar with mean values which are subtracted from channels. 

Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.

scalefactor->multiplier for image values.

swapRB->flag which indicates that swap first and last channels in 3-channel image is necessary.

crop->flag which indicates whether image will be cropped after resize or not

ddepth->Depth of output blob. Choose CV_32F or CV_8U.
"""
# img_blob.shape

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
          "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
          "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
          "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
          "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
          "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
          "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
          "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
          "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
          "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

colors = ['220,20,60', '255,165,0', '128,128,0', '124,252,0', 
          '102,205,170', '25,25,112', '75,0,130', '255,0,255', '119,136,153']
colors = [np.array(color.split(',')).astype('int') for color in colors ]

color = np.array(colors)
# print(colors)
colors = np.tile(colors, (18, 1))
# The numpy.tile() function constructs a new array by repeating array – ‘arr’, the number of times we want to repeat as per repetitions.
# print(colors)

model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

layers = model.getLayerNames()
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.3:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x ,box_center_y, box_width, box_height) = bounding_box.astype('int')
            
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))
            
            end_x = int(start_x + box_width)
            end_y = int(start_y + box_height)
            
            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)


    cv2.imshow('Detection Windows', img)