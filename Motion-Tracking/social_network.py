# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:40:53 2024

@author: Arja Mentink
"""

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np 
import tensorflow as tf
import cv2
from collections import defaultdict
import math
import os
from ultralytics import YOLO
import json


def evaluate_track_ids(detection_model):
    """Function that uses the trained detection and identification model to track and identify the macaques.
        It ensures that no two macaques are given the same identity, and it counts how often each macaques is seen.
        Furthermore, it checks which macaques are in proximity in order to create a social network."""
    
    model = tf.keras.models.load_model("identification_macaques_ft.keras")
    
    frame_number = 0
    iteration=0
    
    count_track_id = defaultdict(lambda: 0)
    count_proximity = defaultdict(lambda: defaultdict(lambda: 0))


    for frame in os.listdir("./frames"):
        
        frame = cv2.imread("./frames/" + frame)
        
        iteration +=1
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = detection_model.predict(frame, conf=0.25, iou=0.6)

        print(frame_number)
        
        # Get the boxes and track IDs
        if results[0].boxes.xywh is not None:
            boxes = results[0].boxes.xywh
            boxes2 = results[0].boxes.xyxy
            
            predictions = get_predictions(boxes, boxes2, frame, model)
    
            # go over every box and get the predictions
            i = 0
            for box, box2 in zip(boxes, boxes2):
                
                prediction = predictions[i]
                
                print("prediction", prediction)
                
                x, y, w, h = box
                x2,y2,x3,y3 = box2
                
                count_track_id[prediction] += 1
                
                #go over the other boxes in the frame and check whom is sitting with whom
                j = 0
                for box_id2, box2_id2 in zip(boxes, boxes2):
                    x4,y4,x5,y5 = box2_id2
                                    
                    prediction2 = predictions[j]
    
                    
                    if not (x2 == x4 and x3 == x5 and y2 == y4 and y3==y5):
                        if in_proximity(box, box_id2, frame_height, frame_width):
                            count_proximity[prediction][prediction2] += 1
                    
                    j+=1
                
                i+=1
            
        frame_number += 1
            

    return count_track_id, count_proximity


def get_predictions(boxes, boxes2, frame, model):
    """Get the predictions of each bounding box in the frame, without conflicts."""
    
    change = True

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    predictions = []
    
    #get the predictions for each frame and save them
    for box, box2 in zip(boxes, boxes2):
        number_conflicts = 0 
        
        x, y, w, h = box
        x2,y2,x3,y3 = box2
                         
        print(frame_width, frame_height, x, y, w, h)
        bounding_box = frame[int(y2):int(y3), int(x2):int(x3)]
        
        bounding_box = cv2.resize(bounding_box, (256,256))
        bounding_box = np.expand_dims(bounding_box, axis=0)

        prediction = model.predict(bounding_box, batch_size = 1)
        
        predictions.append(prediction)


    # if there are more boxes then monkeys, allow doubles
    if len(boxes) > 12:
        max_pred_list = []
        max_pred = defaultdict(lambda: [])
        
        for pred in predictions:
            max_pred_list.append(np.argmax(pred))    
    
    else: #otherwise, continue reducing conflicts till there are none left.
        while change == True: 
            change = False
            
            #for each of the predictions, get the max out and save it, also save to which bb it belongs.
            #repeat this until there is no change to the predictions made. 
            max_pred_list = []
            max_pred = defaultdict(lambda: [])
            
            for pred in predictions:
                #save the predicted label for each box
                max_pred_list.append(np.argmax(pred))    
              
            #check if there are duplicate labels  
            if len(max_pred_list) != len(set(max_pred_list)):
                #if there is, find which ones
                max_pred_list2 = max_pred_list
                
                for i, pred in enumerate(max_pred_list):
                    for j, pred2 in enumerate(max_pred_list2):
                        if pred == pred2 and i != j and pred!= 13:
                            best_conf_1 = max(predictions[i][0])
                            best_conf_2 = max(predictions[j][0])
                            
                            change = True
                            #if the confidence of the second bb is higher, set the corresponding index of the predictions to zero
                            if best_conf_1 <= best_conf_2:
                                predictions[i][0][pred] = 0
                            else:
                                predictions[j][0][pred2] = 0
                    
    return max_pred_list

def true_sn():
    """Create a social network based on the annotated data"""

    label_path = "./labels/all"
    image_path = "./images/all"
    
    count_track_id = defaultdict(lambda: 0)
    count_proximity = defaultdict(lambda: defaultdict(lambda: 0))
    i=0
    for annotation in os.listdir(label_path):
        i+=1
        boxes = open(label_path + annotation, "r")
        
        annotation_jpg = os.path.splitext(annotation)[0] + '.jpg'
            
        image_file = cv2.imread(image_path + annotation_jpg)
        image_height = image_file.shape[0]
        image_width = image_file.shape[1]

        for box in boxes:
            box_parts = box.split()
            #get the right coordinates of the bounding boxes
            label = int(box_parts[0])
            x = int(float(box_parts[1]) * image_width)
            y = int(float(box_parts[2])* image_height)
            w = int(float(box_parts[3]) * image_width) 
            h = int(float(box_parts[4])* image_height) 
            
            #translate the labels to alphanumerical order that EfficientNet uses. 
            if label > 9:
                label = label - 8 
            elif 1 < label < 10:
                label = label + 2
                
            count_track_id[int(label)] += 1
            
            box_coord = x, y, w, h

            boxes2 = open(label_path + annotation, "r")
            # go over every box other box in the frame and calculate the distance
            for box2 in boxes2:
                box_parts2 = box2.split()
                #get the right coordinates of the bounding boxes
                label2 = int(box_parts2[0])
                x2 = int(float(box_parts2[1]) * image_width)
                y2 = int(float(box_parts2[2])* image_height)
                w2 = int(float(box_parts2[3]) * image_width) 
                h2 = int(float(box_parts2[4])* image_height) 
                
                box2_coord = x2, y2, w2, h2
                
                if label2 > 9:
                    label2 = label2 - 8 
                elif 1 < label2 < 10:
                    label2 = label2 + 2
                     
                if label != label2:
                    if in_proximity(box_coord, box2_coord, image_height, image_width):
                        count_proximity[int(label)][int(label2)] += 1
                

    return count_track_id, count_proximity

def in_proximity(box1, box2, frame_height, frame_width):
    """Checks whether two boxes are in close proximity in the 3D-space."""
    
    x,y,w,h = box1
    x = float(x)
    y = float(y)
    x2,y2,w2,h2 = box2
    x2 = float(x2)
    y2 = float(y2)
    
    point1 = [x,y]
    point2 = [x2,y2]
    
    ##Orange platform
    #for each area, check whether both centres lie within the area and are close together
    ctr1 = np.array([[1200,500], [1200,1050], [frame_width, 500], [frame_width, 1050]]).reshape((-1,1,2)).astype(np.int32)
    
    if cv2.pointPolygonTest(ctr1, (x,y), False) >= 0.0 and cv2.pointPolygonTest(ctr1, (x2,y2), False) >= 0.0:
        if math.dist(point1, point2) < 250: #threshold of distance depends on area, areas further away have smaller threshold. 
            return True
        
    ##Window in the front
    ctr2 = np.array([[0,200], [0,frame_height], [500, 200], [500, frame_height]]).reshape((-1,1,2)).astype(np.int32)
    
    if cv2.pointPolygonTest(ctr2, (x,y), False) >= 0.0 and cv2.pointPolygonTest(ctr2, (x2,y2), False) >= 0.0:
        if math.dist(point1, point2) < 300: #threshold of distance depends on area, areas further away have smaller threshold. 
            return True
      
    ##Ground
    ctr3 = np.array([[800,1050], [800,frame_height], [1600, 1050], [1600, frame_height]]).reshape((-1,1,2)).astype(np.int32)
    
    if cv2.pointPolygonTest(ctr3, (x,y), False) >= 0.0 and cv2.pointPolygonTest(ctr3, (x2,y2), False) >= 0.0:
        if math.dist(point1, point2) < 150: #threshold of distance depends on area, areas further away have smaller threshold. 
            return True
        
    
    ##Wooden walkway
    ctr4 = np.array([[500,500], [500,800], [750, 500], [750, 800]]).reshape((-1,1,2)).astype(np.int32)
    
    if cv2.pointPolygonTest(ctr4, (x,y), False) >= 0.0 and cv2.pointPolygonTest(ctr4, (x2,y2), False) >= 0.0:
        if math.dist(point1, point2) < 125: #threshold of distance depends on area, areas further away have smaller threshold. 
            return True
        
        
    ##Window sil
    ctr5 = np.array([[750,300], [750,700], [1100, 300], [1100, 700]]).reshape((-1,1,2)).astype(np.int32)
    
    if cv2.pointPolygonTest(ctr5, (x,y), False) >= 0.0 and cv2.pointPolygonTest(ctr5, (x2,y2), False) >= 0.0:
        if math.dist(point1, point2) < 75 : #threshold of distance depends on area, areas further away have smaller threshold. 
            return True
        
    return False


#Get best identification model
detection_model = YOLO('./best_detect_macaques.pt')

count_track_id, count_proximity = evaluate_track_ids(detection_model)

#count_track_id, count_proximity = true_sn(detection_model, test_video, test_dir)


file = "./count_track_id.json"
with open(file, 'w') as f: 
    json.dump(count_track_id, f, default=str)

file = "./count_proximity.json"
with open(file, 'w') as f: 
    json.dump(count_proximity, f, default=str)