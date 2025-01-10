dataset_type = 'NuScenesMonoDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    # dict(
    #     type='LoadAnnotations3D',
    #     with_bbox=True,
    #     with_label=True,
    #     with_attr_label=True,
    #     with_bbox_3d=True,
    #     with_label_3d=True,
    #     with_bbox_depth=True),
    dict(type='Resize', img_scale=(384, 192), keep_ratio=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=8),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='Resize', img_scale=(384, 192), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img'
        ]),
    # dict(
    #     type='MultiScaleFlipAug',
    #     scale_factor=1.0,
    #     flip=False,
    #     transforms=[
    #         dict(type='RandomFlip3D'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['img']),
    #     ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=2)

seed = 0

# Optimizer
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    # lr=5e-6,
    betas=(0.9, 0.95),  # the momentum is change during training
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "bias": dict(decay_mult=0.0),
        }
    ),
    weight_decay=0.0001,
)
log_steps = 50
checkpointing_steps = 1000

