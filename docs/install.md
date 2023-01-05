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
sudo docker run (--gpus all) --shm-size=8g –it –v —name {CONTAINER_NAME} \
{WORK_DIR}:/mmdetection cbnuirl/swin_transf_cbnu:1.0
```

--gpus all for multi-GPU. Exclude when using single-GPU.
If you want to change into container bash:
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

Then, run:
```
pip install tqdm # If not installed
python tools/dataset_converters/morai.py --data_path {DATA_PATH}
```

{DATA_PATH} be like data/(DATA_NAME). label folder can be deleted.

# Change Configuration File
