# The new config inherits a base config to highlight the necessary modification
_base_ = 'hybrid_task_cascade_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation

# Modify dataset related settings
data_root = 'data/morai_coco_mix/'
data = dict(
    train=dict(
        img_prefix=data_root + 'train',
        ann_file='data/morai_coco_mix/annotations/train.json'),
    val=dict(
        img_prefix=data_root + 'val',
        ann_file='data/morai_coco_mix/annotations/val.json'),
    test=dict(
        img_prefix=data_root + 'val',
        ann_file='data/morai_coco_mix/annotations/val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
