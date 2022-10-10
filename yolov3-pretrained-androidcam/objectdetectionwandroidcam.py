# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:12:52 2022

@author: S BASA
"""

import cv2 
import numpy as np
import requests

url = 'http://192.168.0.187:8080/shot.jpg'

while True:
    
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(img, (640, 480))
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
    
    

    
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

    colors = np.tile(colors, (18, 1))

    
    model = cv2.dnn.readNetFromDarknet('YOLO/yolov3-pretrained-model/yolov3.cfg', 'C:/Users/S BASA/Desktop/yolov3.weights')
    
    layers = model.getLayerNames()
    output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)
    
    # non maximum suppression
    id_list = []
    boxes_list = []
    confidence_list = []
    
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.4:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x ,box_center_y, box_width, box_height) = bounding_box.astype('int')
                
                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))
                
                # non maximum suppression
                id_list.append(predicted_id)
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                confidence_list.append(float(confidence))
    
    # non maximum suppression            
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)

    max_class_id = max_ids[0]
    box = boxes_list[max_class_id]
    
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height =  box[3]
    
    predicted_id = id_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidence_list[max_class_id]
    
    
    end_x = int(start_x + box_width)
    end_y = int(start_y + box_height)
    
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
    
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
    cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    cv2.imshow('Android Cam', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break     
        
    
    

cv2.destroyAllWindows()