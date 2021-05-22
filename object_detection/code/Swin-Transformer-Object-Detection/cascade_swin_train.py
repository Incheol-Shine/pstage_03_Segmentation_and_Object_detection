#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


# In[2]:


import wandb
project_name = "object_detection"
wandb.init(project=project_name, reinit=True)

wandb.run.name = 'cascade_swin_fold1.ipynb'
# generted run ID로 하고 싶다면 다음과 같이 쓴다.
# wandb.run.name = wandb.run.id
wandb.run.save()


# In[3]:


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
cfg = Config.fromfile('./configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py')

PREFIX = '../../input/data/'


# In[4]:


print(cfg.dump())


# In[5]:


cfg.log_config.hooks[1].init_kwargs.project = project_name
cfg.log_config.hooks[1].init_kwargs.name = wandb.run.name


# In[6]:


##### albu_augmentation
albu_train_transforms = [
    dict(
        type='Resize',
        width=512,
        height=512,
        p=1.0),
    dict(
        type='Rotate',
        limit=10,
        p=0.5), 
    dict(
        type='RandomBrightnessContrast',
        p=1.0),
]
#####


# In[7]:


# img_norm_cfg
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train.json'
# cfg.data.train.pipeline[2]['img_scale'] = (512, 512)
# cfg.data.train.pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='Pad', size_divisor=32),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels'],
#         meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
#                    'pad_shape', 'scale_factor')
#     )
# ]

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'
# cfg.data.val.pipeline[1]['img_scale'] = (512, 512)


cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
# cfg.data.test.pipeline[1]['img_scale'] = (512, 512)


# num_workers?
cfg.data.samples_per_gpu = 4

# seed
cfg.seed=2020

# ??
cfg.gpu_ids = [0]

# 새롭게 만든 모델을 저장할 장소
cfg.work_dir = './work_dirs/cascade_mask_rcnn_swin_1x_trash'

# class 개수 11개로 수정
cfg.model.roi_head.bbox_head[0].num_classes=11
cfg.model.roi_head.bbox_head[1].num_classes=11
cfg.model.roi_head.bbox_head[2].num_classes=11
cfg.model.roi_head.mask_head.num_classes =11

# ??
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)


# In[8]:


model = build_detector(cfg.model)


# In[9]:


datasets = [build_dataset(cfg.data.train)]


# In[10]:


train_detector(model, datasets[0], cfg, distributed=False, validate=True)


# In[ ]:




