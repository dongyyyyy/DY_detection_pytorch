import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import json
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def read_text_file_txt(file_path,image_name,image_shape):# YOLO Label을 읽기 위한 함수
    width, height = image_shape
    file_path = file_path+image_name+".txt"
    print(file_path)
    f = open(file_path,"r")
    label = []
    while True:
        line = f.readline() # 줄별로 읽기
        if not line: break # 끝
        line_label = line.split(' ') # 공백 기준으로 문자열 자르기
        length = len(line_label) # label 데이터 문자열 길이
        if(length == 5): # YOLO label dataset
            label_x = float(line_label[1])
            label_y = float(line_label[2])
            label_w = float(line_label[3])
            label_h = float(line_label[4])
            label.append({"classes": line_label[0], "x": label_x, "y":label_y,"width":label_w,"height":label_h})

    return label # return은 map형식으로 return됨. ( classes , x , y , width , height ) 순

def read_text_file(image_name,image_shape):# YOLO Label을 읽기 위한 함수
    dataDir = './../../../COCO_dataset'
    dataType = 'val2017'
    annFile = '{}/annotation/instances_{}.json'.format(dataDir,dataType)
    print(annFile)
    width , height = image_shape
    label = []
    with open(annFile) as json_file:
        json_data = json.load(json_file)
        json_image = json_data["images"]
        for i in json_image:
            if(i["file_name"]==image_name):
                id = i["id"]
        json_bbox = json_data["annotations"]
        for i in json_bbox:
            if(i["image_id"]==id):
                bbox = i["bbox"]
                print("bbox:",bbox) # x ,y , width , height
                for i in range(0,len(bbox),4):
                    label_x = (bbox[i] + bbox[i+2]/2)/width
                    label_y = (bbox[i+1] + bbox[i+3]/2)/height
                    label_w = bbox[i+2]/width
                    label_h = bbox[i+3]/height
                    label.append({"classes": 0, "x": label_x, "y":label_y,"width":label_w,"height":label_h})
    return label

class ImageDataset_withLabel(Dataset):
    def __init__(self, root, resize_shape, root_label):
        self.root = root+'\\'
        resize_height, resize_width = resize_shape
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize((resize_height,resize_width), Image.NEAREST), # image resize
                transforms.ToTensor(), # float to tensor
                transforms.Normalize(mean,std), # Normalize
            ]
        )

        self.image_files = sorted(glob.glob(root+"/*.jpg")) # image sort
        self.label_path = root_label
        self.label_files = sorted(glob.glob(root_label+"/*.txt")) # label sort

    def __getitem__(self,index):
        img = Image.open(self.image_files[index % len(self.image_files)])
        print(self.root)
        img_name, _ = os.path.splitext(self.image_files[index%len(self.image_files)].replace(self.root,""))
        print("img : ",img_name)
        origin_width , origin_height = img.size
        origin_img = np.array(img.getdata()).reshape(img.size[0], img.size[1],-1)

        img_data = self.resize_transform(img)
        label = read_text_file_txt(self.label_path,img_name,[origin_width,origin_height])

        return {"img": img_data, "label": label, "origin_width":origin_width,"origin_height":origin_height,"origin_img":origin_img}

    def __len__(self):  # data size를 넘겨주는 파트
        return len(self.image_files)  # 파일 길이 반환 ( 총 이미지 수 )

