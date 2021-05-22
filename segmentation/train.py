import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')


from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

#set() 시각화를 위한 라이브러리

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장
batch_size = 8   # Mini-batch size
num_epochs = 20
learning_rate = 0.0001

from dataset import get_data
train_loader, val_loader, test_loader = get_data()


# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
import segmentation_models_pytorch as smp

# import wandb
# wandb.init(project="stage3-seg-0")

model = smp.DeepLabV3(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="swsl",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=12,                      # model output channels (number of classes in your dataset)
)
# wandb.config.epoch = 20
# wandb.config.learning_rate = 0.0001
# wandb.config.batch_size = 8


x = torch.randn([2, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x).to(device)
print("output shape : ", out.size())

model = model.to(device)
class_labels = {0:'Backgroud', 1:'UNKNOWN', 2:'General trash', 3:'Paper', 4:'Paper pack', 5:'Metal', 6:'Glass', 7:'Plastic', 8:'Styrofoam', 9:'Plastic bag', 10:'Battery', 11:'Clothing'}
"""## train, validation, test 함수 정의"""

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    # best_loss = 9999999
    best_miou = 0

    # wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(num_epochs):
        hist = np.zeros((12, 12))
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                # wandb.log({"epoch": epoch, "loss": loss}, step=step)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item()))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        # if (epoch + 1) % val_every == 0:
        #     avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
        #     if avrg_loss < best_loss:
        #         print('Best performance at epoch: {}'.format(epoch + 1))
        #         print('Save model in', saved_dir)
        #         best_loss = avrg_loss
        #         save_model(model, saved_dir)
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_miou = validation(epoch + 1, model, val_loader, criterion, device)
            if val_miou > best_miou:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_miou = val_miou
                save_model(model, saved_dir)

from utils import add_hist
def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((12, 12))

    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            # for i in range(8):
            #     wandb.log(wandb.Image(images, masks={
            #         "predictions" : {
            #             "mask_data" : outputs[i],
            #             "class_labels" : class_labels
            #         },
            #         "ground_truth" : {
            #             "mask_data" : masks[i],
            #             "class_labels" : class_labels
            #         }
            #     }))
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)

            # mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # mIoU_list.append(mIoU)
        

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
        avrg_loss = total_loss / cnt
        # wandb.log({"epoch": epoch, "loss": loss, "mIoU": np.mean(mIoU_list)})
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mIoU))

    return avrg_loss, mIoU

"""## 모델 저장 함수 정의"""

# 모델 저장 함수 정의
val_every = 1

saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
    
def save_model(model, saved_dir, file_name='DeepLabv3_best_model(vgg16).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

"""## 모델 생성 및 Loss function, Optimizer 정의"""

# Loss function 정의
criterion = nn.CrossEntropyLoss()

# Optimizer 정의
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)

"""## 저장된 model 불러오기 (학습된 이후) """

# best model 저장된 경로
model_path = './saved/DeepLabv3_best_model(vgg16).pt'

# best model 불러오기
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)

# 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
# model.eval()

# 첫번째 batch의 추론 결과 확인
for imgs, image_infos in test_loader:
    image_infos = image_infos
    temp_images = imgs
    
    model.eval()
    # inference
    outs = model(torch.stack(temp_images).to(device))
    oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
    
    break



"""## submission을 위한 test 함수 정의"""

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

"""## submission.csv 생성"""

# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

# test set에 대한 prediction
file_names, preds = test(model, test_loader, device)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv("./submission/Baseline_DeepLabv3(vgg16).csv", index=False)