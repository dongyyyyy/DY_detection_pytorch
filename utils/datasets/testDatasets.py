from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from utils.datasets.read_coco_dataset import *
import cv2

coco_classes= {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
               9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
               16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
               24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
               34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
               40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
               47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
               56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
               64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
               75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
               82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
#print(read_text_file("./../../test.txt"))

if __name__ == '__main__':
    dataloader = DataLoader(  # training data read
            ImageDataset_withLabel(root="../../../COCO_dataset/train2017" , resize_shape=(416,416),root_label="../../../COCO_dataset/annotation/train2017/"),# root = ../../data/img_align_celeba &  hr_shape = hr_shape
            batch_size=1,  # batch size ( mini-batch )
            shuffle=True,  # shuffle
            num_workers=1, # using 8 cpu threads
        )

    for i, dataset in enumerate(dataloader):
        #print("i:{} ==> img.shape :{},\nlabel :{}".format(i,dataset["img"],dataset["label"]))
        #print(len(dataset["label"]))
        #기존 이미지 불러오기
        origin_img = dataset["origin_img"]
        #print("origin_img's shape : ",origin_img.shape)
        origin_img = np.squeeze(origin_img, axis=0) # 차원 삭제 axis = 0 ==> 가장 맨 앞의 배열 차원을 없앤다.
        #print("origin_img's shape : ", origin_img.shape)
        origin_img = np.array(origin_img, dtype='uint8') # pil저장 시 자료형태는 uint8 or float이여야 한다.
        origin_img = np.reshape(origin_img,[dataset["origin_height"],dataset["origin_width"],-1])
        #plt.imshow(origin_img)
        #plt.show()
        for j in dataset["label"]:
            print(j) # 전체 map 정보 출력
            print('classes : {}'.format(j["classes"])) # 각 key를 통한 value 접근

            #현재 크기에 맞게 label값 변경
            x = float(j["x"][0]) * dataset["origin_width"]
            y = float(j["y"][0]) * dataset["origin_height"]
            width = float(j["width"][0]) * dataset["origin_width"]
            height = float(j["height"][0]) * dataset["origin_height"]

            print("x : {}, y : {}, width : {}, height : {}".format(x,y,width,height))
            #rectangle을 그리기 위한 x1,y1 & x2,y2 값
            x1 = x - width/2
            y1 = y - height/2
            x2 = x + width/2
            y2 = y + height/2
            class_name = coco_classes.get(int(j.get("classes")[0]))
            cv2.putText(origin_img,class_name,(x1,y1+10),cv2.FONT_ITALIC,0.3,(0,0,255),1)
            cv2.rectangle(origin_img, (x1,y1), (x2,y2), (0, 0, 255), 1) # label을 통하여 boundingBox그리기
            # cv.rectangle(img,(x1,y1),(x2,y2),color,thickness,lineType,shift)
        #save_image(dataset["img"],"./%d.jpg"%i)
        #cv shape = (height, width, channel)
        plt.imshow(origin_img)
        plt.show()
        plt.imsave('./newImage.jpg',origin_img)