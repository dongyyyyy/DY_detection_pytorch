import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def read_text_file(file_path,index):# YOLO Label을 읽기 위한 함수
    file_path = file_path[index]
    print(file_path)
    f = open(file_path,"r")
    label = []
    while True:
        line = f.readline() # 줄별로 읽기
        if not line: break # 끝
        line_label = line.split(' ') # 공백 기준으로 문자열 자르기
        length = len(line_label) # label 데이터 문자열 길이
        if(length == 5): # YOLO label dataset
            label.append({"classes": line_label[0], "x": line_label[1], "y":line_label[2],"width":line_label[3],"height":line_label[4].strip('\n')})

    return label # return은 map형식으로 return됨. ( classes , x , y , width , height ) 순



class ImageDataset_withLabel(Dataset):
    def __init__(self, root, resize_shape, root_label):
        resize_height, resize_width = resize_shape
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize((resize_height,resize_width), Image.NEAREST), # image resize
                transforms.ToTensor(), # float to tensor
                transforms.Normalize(mean,std), # Normalize
            ]
        )

        self.image_files = sorted(glob.glob(root+"/*.jpg")) # image sort
        self.label_files = sorted(glob.glob(root_label+"/*.txt")) # label sort

    def __getitem__(self,index):
        img = Image.open(self.image_files[index % len(self.image_files)])
        origin_width , origin_height = img.size
        origin_img = np.array(img.getdata()).reshape(img.size[0], img.size[1],-1)

        img_data = self.resize_transform(img)
        label = read_text_file(self.label_files,index % len(self.image_files))

        return {"img": img_data, "label": label, "origin_width":origin_width,"origin_height":origin_height,"origin_img":origin_img}

    def __len__(self):  # data size를 넘겨주는 파트
        return len(self.image_files)  # 파일 길이 반환 ( 총 이미지 수 )

