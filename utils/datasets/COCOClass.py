import json
import os

def read_text_file():# YOLO Label을 읽기 위한 함수
    dataDir = './../../../COCO_dataset'
    dataType = 'val2017'
    annFile = '{}/annotation/instances_{}.json'.format(dataDir,dataType)
    print(annFile)
    #width , height = image_shape
    #label = []
    #i = 0
    with open(annFile) as json_file:
        json_data = json.load(json_file)
        json_image = json_data["categories"]
        #print(json_image)
        label = {}
        for categories in json_image:
            name = categories["name"]
            id = categories["id"]
            label[id] = name
        print(label)
    #return label

read_text_file()

