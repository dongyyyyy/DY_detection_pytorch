from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from utils.datasets.readDatasets import *
import cv2

#print(read_text_file("./../../test.txt"))

if __name__ == '__main__':
    dataloader = DataLoader(  # training data read
            ImageDataset_withLabel(root="../../" , resize_shape=(416,416),root_label="../../"),# root = ../../data/img_align_celeba &  hr_shape = hr_shape
            batch_size=1,  # batch size ( mini-batch )
            shuffle=True,  # shuffle
            num_workers=1, # using 8 cpu threads
        )

    for i, dataset in enumerate(dataloader):
        print("i:{} ==> img.shape :{},\nlabel :{}".format(i,dataset["img"],dataset["label"]))
        print(len(dataset["label"]))
        #기존 이미지 불러오기
        origin_img = dataset["origin_img"]
        print("origin_img's shape : ",origin_img.shape)
        origin_img = np.squeeze(origin_img, axis=0) # 차원 삭제 axis = 0 ==> 가장 맨 앞의 배열 차원을 없앤다.
        print("origin_img's shape : ", origin_img.shape)
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

            cv2.rectangle(origin_img, (x1,y1), (x2,y2), (0, 0, 255), 1) # label을 통하여 boundingBox그리기
            # cv.rectangle(img,(x1,y1),(x2,y2),color,thickness,lineType,shift)
        save_image(dataset["img"],"./%d.jpg"%i)
        #cv shape = (height, width, channel)
        plt.imshow(origin_img)
        plt.show()
        plt.imsave('./newImage.jpg',origin_img)