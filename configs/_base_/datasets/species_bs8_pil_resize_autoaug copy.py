_base_ = [
    'pipelines/auto_aug.py',
]

# dataset settings
dataset_type = 'Species'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/media/hexaburbach/onetb/hexa_train_dataset/mmcls/species/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/media/hexaburbach/onetb/hexa_train_dataset/mmcls/species/val',
        ann_file='/media/hexaburbach/onetb/hexa_train_dataset/mmcls/species/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/media/hexaburbach/onetb/hexa_train_dataset/mmcls/species/val',
        ann_file='/media/hexaburbach/onetb/hexa_train_dataset/mmcls/species/meta/val.txt',
        pipeline=test_pipeline))


# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(interval=20, save_best="auto", metric='accuracy', metric_options={'topk': (1, 5)})
