from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.image as image
import matplotlib.pyplot as plt
from utils.datasets.readDatasets import *

#print(read_text_file("./../../test.txt"))
if __name__ == '__main__':
    dataloader = DataLoader(  # training data read
            ImageDataset_withLabel(root="../../" , resize_shape=(416,416),root_label="../../"),# root = ../../data/img_align_celeba &  hr_shape = hr_shape
            batch_size=1,  # batch size ( mini-batch )
            shuffle=True,  # shuffle
            num_workers=1,  # using 8 cpu threads
        )

    for i, dataset in enumerate(dataloader):
        print("i:{} ==> img.shape :{}, label :{}",i,dataset["img"])
        print(len(dataset["label"]))
        for j in dataset["label"]:
            print(j) # 전체 map 정보 출력
            print('classes : {}'.format(j["classes"])) # 각 key를 통한 value 접근
        save_image(dataset["img"],"./%d.jpg"%i)