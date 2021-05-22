# py 파일 실행할 때 불러옴
_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',   ## mmdet 에 기존 작성되어있는 모델
    '../dataset.py',                                ## 쓰레기 데이터셋
    '../../_base_/schedules/schedule_1x.py',        ## 기존 작성된 스케줄러
    '../../_base_/default_runtime.py'               ## 기존 작성된 ?? 
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=11    ## 결국 num_classes 하나를 수정하기 위해 이렇게 코드를 넣어야 한다.
        )
    )
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")