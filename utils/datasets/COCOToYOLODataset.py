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
        json_image = json_data["images"]
        #print(json_image)

        for image_info in json_image:
            file_name = image_info["file_name"]
            id = image_info["id"]
            width = image_info["width"]
            height = image_info["height"]
            #print("id : {} width : {} height : {}".format(id,width,height))
            json_bbox = json_data["annotations"]
            write_file_name, _ = os.path.splitext(file_name)
            f = open("./../../../COCO_dataset/annotation/val2017/{}.txt".format(write_file_name), 'w')
            for i in json_bbox:
                if(i["image_id"]==id):
                    bbox = i["bbox"]
                    classes = i["category_id"]
                #print("bbox:",bbox) # x ,y , width , height
                    for j in range(0,len(bbox),4):
                        label_x = (bbox[j] + bbox[j+2]/2)/width
                        label_y = (bbox[j+1] + bbox[j+3]/2)/height
                        label_w = bbox[j+2]/width
                        label_h = bbox[j+3]/height
                        data = "{} {} {} {} {}\n".format(classes,label_x,label_y,label_w,label_h)
                        f.write(data)
            f.close()

    #return label

read_text_file()

