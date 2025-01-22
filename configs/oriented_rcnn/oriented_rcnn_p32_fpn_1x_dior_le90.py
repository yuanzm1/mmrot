_base_ = [
    './oriented_rcnn_r50_fpn_1x_dior_le90.py'
]

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
angle_version = 'le90'
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.VisionTransformer',
        arch='b',
        output_cls_token=False,
        img_size=1024,
        patch_size=32,
        drop_rate=0.,
        layer_cfgs=dict(act_cfg=dict(type='GELU')),
        init_cfg=dict(type='Pretrained', checkpoint="/home/yuanzm/mmdetection/weights/mmpre-RemoteCLIP-ViT-B-32.pt", prefix='visual.'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[768],
        out_channels=256,
        num_outs=1),
    rpn_head=dict(
        anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8, 16, 32],
        ratios=[0.5, 1.0, 2.0],
        strides=[32])),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            featmap_strides=[32]),
        bbox_head=dict(
            num_classes=20,
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(lr=0.005)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=6)