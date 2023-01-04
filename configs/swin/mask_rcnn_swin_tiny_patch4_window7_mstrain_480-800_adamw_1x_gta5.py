# The new config inherits a base config to highlight the necessary modification
_base_ = 'mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8),
        mask_head=dict(num_classes=8)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
data = dict(
    train=dict(
        img_prefix='data/gta5/images/train',
        classes=classes,
        ann_file='data/gta5/annotations/gta5_train.json'),
    val=dict(
        img_prefix='data/gta5/images/val',
        classes=classes,
        ann_file='data/gta5/annotations/gta5_val.json'),
    test=dict(
        img_prefix='data/gta5/images/test',
        classes=classes,
        ann_file='data/gta5/annotations/gta5_test.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'