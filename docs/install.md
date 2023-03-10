# Environment setting

## Install Docker

### Install with script

```
sudo wget -qO- https://get.docker.com/ | sh
```
or
```
sudo curl -fsSL https://get.docker.com/ | sh
```

## Git Clone

```
git clone https://github.com/cbnuirl/cbnu_swin-transformer.git
cd cbnu_swin-transformer
```

## Docker Image

### Compressed file(if exists)

```
sudo docker load -i swin_transf_cbnu.tar
```

### dockerfile

```
sudo docker build –t cbnuirl/swin_transf_cbnu:1.0 docker/
```

## Docker Container

```
sudo docker run --gpus all --name {CONTAINER_NAME} --shm-size=8g –it –v \
{WORK_DIR}:/mmdetection/ cbnuirl/swin_transf_cbnu:1.0
```

WORK_DIR is the path of cbnu_swin-transformer(/path/to/cbnu_swin-transformer)

***NOTE***: If error, install nvidia-container-toolkit

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Install other requirements

```
pip install timm
pip uninstall pycocotools
pip install mmpycocotools
```

***NOTE***:If you want to change into container bash:
```
sudo docker start {CONTAINER_NAME}
sudo docker attach {CONTAINER_NAME}
```

# Data Conversion

Change into COCO format like:
```
data
	└(DATA_NAME)/
		├annotations/
		│	├train.json -> COCO format annotation file
		│	└val.json
		├train/
		│	└XX_XX_TXWX_XX_XXX_REXX_XXX.png -> original image
		└val/
			└XX_XX_TXWX_XX_XXX_REXX_XXX.png
```

Prepare data like:
```
data
	└(DATA_NAME)/
		├label/
		│	└XX_XX_TXWX_XX_XXX_REXX_XXX.json
		│	-> .json file includes bounding box
		├train/
		│	└XX_XX_TXWX_XX_XXX_REXX_XXX.png -> original image
		└val/
			└XX_XX_TXWX_XX_XXX_REXX_XXX.png
```

Change tools/dataset_converters/morai.py if categories are different:
```
…
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
…
```

Change mmdet/datasets/coco.py CLASSES:
```
…
CLASSES = ('vehicle', 'bus', 'truck', 'policeCar', 'ambulance',
		'schoolBus', 'otherCar', 'motorcycle', 'bicycle', 'twoWheeler',
		'pedestrian', 'rider', 'freespace', 'curb', 'sidewalk',
		'crossWalk', 'safetyZone', 'speedBump', 'roadMark', 'whiteLane',
		'yellowLane', 'blueLane', 'redLane', 'stopLane', 'trafficSign',
		'trafficLight', 'constructionGuide', 'trafficDrum', 'rubberCone', 'warningTriangle',
		'fence', 'egoVehicle', 'background')
…
```

Then, run:
```
pip install tqdm # If not installed
```
```
python tools/dataset_converters/morai.py --data_path {DATA_PATH}
```

{DATA_PATH} would be like data/(DATA_NAME). label folder can be deleted.

# Change Configuration File

Create new configuration file for new data. Location is configs/swin/.
Save like 'cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_{DATA_NAME}.py'
Revise content:
```
…
# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
…
data_root = ‘data/{DATA_NAME}/’
…
data = dict(
    samples_per_gpu=2, <- batch size
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
…
```

For mixed training:
```
…
dataset_A_train=dict(
    type=dataset_type,
    ann_file=data_root + 'synthetic/' + 'annotations/train.json',
    img_prefix=data_root + 'synthetic/' + 'train/',
    pipeline=train_pipeline)

dataset_B_train=dict(
    type=dataset_type,
    ann_file=data_root + 'real/' + 'annotations/train.json',
    img_prefix=data_root + 'real/' + 'train/',
    pipeline=train_pipeline)

dataset_B_test=dict(
    type=dataset_type,
    ann_file=data_root + 'real/' + 'annotations/val.json',
    img_prefix=data_root + 'real/' + 'val/',
    pipeline=test_pipeline)

dataset_B_val=dict(
    type=dataset_type,
    ann_file=data_root + 'real/' + 'annotations/val.json',
    img_prefix=data_root + 'real/' + 'val/',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
	dataset_A_train,
	dataset_B_train
    ],
    val=dataset_B_val,
    test=dataset_B_test)
…
```

Mixed training data should be distributed like:
```
data
	└(DATA_NAME)/
		├real/
		│	├annotations/
		│	├train/
		│	└val/
		└synthetic/
			├annotations/
			├train/
			└(val/)
```
Validation set that will not be used can be removed.

If you want to use pre-trained model to continue training or fine-tuning, add configuration file:
```
_base_ = [
…
]
load_from = “(PRETRAINED_MODEL)”
…
```
PRETRAINED_MODEL will be like "checkpoints/cascade…morai_daegu.pth"
