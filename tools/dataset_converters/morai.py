import os
from tqdm import tqdm
import json
import argparse
# import cv2

parser = argparse.ArgumentParser(description="Convert MORAI into COCO")
parser.add_argument("--data_path", type=str, help="MORAI data path")
args = parser.parse_args()
if args.data_path[len(args.data_path) - 1] != '/':
    args.data_path = args.data_path + '/'

img_path = [args.data_path + "train/", args.data_path + "val/", args.data_path + "test/"]
label_path = args.data_path + "label/"

if not os.path.isdir(args.data_path + "annotations/"):
    os.mkdir(args.data_path + "annotations/")

classes = []
for split in img_path:
    if not os.path.isdir(split):
        continue
    img_list = os.listdir(split)
    img_list.sort()

    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "vehicle"},
        {"id": 2, "name": "bus"},
        {"id": 3, "name": "truck"},
        {"id": 4, "name": "policeCar"},
        {"id": 5, "name": "ambulance"},
        {"id": 6, "name": "schoolBus"},
        {"id": 7, "name": "otherCar"},
        {"id": 8, "name": "motorcycle"},
        {"id": 9, "name": "bicycle"},
        {"id": 10, "name": "twoWheeler"},
        {"id": 11, "name": "pedestrian"},
        {"id": 12, "name": "rider"},
        {"id": 13, "name": "freespace"},
        {"id": 14, "name": "curb"},
        {"id": 15, "name": "sidewalk"},
        {"id": 16, "name": "crossWalk"},
        {"id": 17, "name": "safetyZone"},
        {"id": 18, "name": "speedBump"},
        {"id": 19, "name": "roadMark"},
        {"id": 20, "name": "whiteLane"},
        {"id": 21, "name": "yellowLane"},
        {"id": 22, "name": "blueLane"},
        {"id": 23, "name": "redLane"},
        {"id": 24, "name": "stopLane"},
        {"id": 25, "name": "trafficSign"},
        {"id": 26, "name": "trafficLight"},
        {"id": 27, "name": "constructionGuide"},
        {"id": 28, "name": "trafficDrum"},
        {"id": 29, "name": "rubberCone"},
        {"id": 30, "name": "warningTriangle"},
        {"id": 31, "name": "fence"},
        {"id": 32, "name": "egoVehicle"},
        {"id": 33, "name": "background"}
    ]

    id_count = 0
    for img in tqdm(img_list):
        label_file = label_path + img[:-4] + ".json"
        with open(label_file, 'r') as f:
            label = json.load(f)
        
        # test
        # frame = cv2.imread(split + img, cv2.IMREAD_COLOR)

        # images
        file_name = label['information']['filename']
        id = int(
            file_name[7] + file_name[9] + file_name[12]
            + file_name[15:17] + file_name[20:22] + file_name[23:26]
        )
        width = label['information']['resolution'][0]
        height = label['information']['resolution'][1]

        image = dict(
            id=id,
            width=width,
            height=height,
            file_name=file_name
        )
        images.append(image)

        # annotations
        annos = label['annotations']
        for ann in annos:
            if ann['distance'] == -1 or ann['bbox'] == []:
                continue
            
            # annotations id is id_count, so skip
            # and annotations image_id is images id
            image_id = id
            # category_id = 1
            # if ann['class'] == "Pedestrian":
            #     category_id = 11
            # elif ann['class'] == "bicycle":
            #     category_id = 9
            # elif ann['class'] == "animal":
            #     continue
            for cat in categories:
                if cat['name'].lower() == ann['class'].lower():
                    category_id = cat['id']

            if ann['class'] not in classes:
                classes.append(ann['class'])
            if ann['class'] == "animal":
                continue

            # no mask information
            segmentation = []
            x1, y1, x2, y2 = (
                ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]
            )
            area = float((x2 - x1) * (y2 - y1))
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            iscrowd = 0

            # frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            annotation = dict(
                id=id_count,
                image_id=image_id,
                category_id=category_id,
                segmentation=segmentation,
                area=area,
                bbox=bbox,
                iscrowd=iscrowd
            )
            annotations.append(annotation)
            id_count += 1
        
        # cv2.imwrite(split + "../frames/" + img, frame)

    created_annotations = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )

    if split == args.data_path + "train/":
        ann_file_name = args.data_path + "annotations/train.json"
    elif split == args.data_path + "val/":
        ann_file_name = args.data_path + "annotations/val.json"
    elif split == args.data_path + "test/":
        ann_file_name = args.data_path + "annotations/test.json"
    with open(ann_file_name, 'w') as f:
        json.dump(created_annotations, f)

print("categories", classes)
