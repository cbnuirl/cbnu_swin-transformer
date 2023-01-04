# The new config inherits a base config to highlight the necessary modification
_base_ = 'hybrid_task_cascade_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation

# Modify dataset related settings
data_root = 'data/morai/'
data = dict(
    train=dict(
        img_prefix=data_root + 'train',
        ann_file='data/morai/annotations/train.json'),
    val=dict(
        img_prefix=data_root + 'val',
        ann_file='data/morai/annotations/val.json'),
    test=dict(
        img_prefix=data_root + 'val',
        ann_file='data/morai/annotations/val.json'))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[7])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=8)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'work_dirs/hybrid_task_cascade_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai/epoch_36.pth'
