import time
import datetime
import os
import copy
import cv2
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import dill
from sklearn.metrics import recall_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hyper_param_batch = 1

random_seed = 100
random.seed(random_seed)
torch.manual_seed(random_seed)

num_classes = 4
model_name = 'efficientnet-b7'

train_name = 'model2'

PATH = './scalp_weights/' #'C:/Users/lee58/OneDrive/Desktop/FileEX/scalp_weights/원천데이터/6.탈모'

#data_train_path = '../train_data/'+train_name+'/.'
#data_validation_path = '../train_data/'+train_name+'/.'
#data_test_path = '../train_data/'+train_name+'/.'

data_train_path = '/content/drive/MyDrive/DATA'
data_validation_path = '//content/drive/MyDrive/DATA'
data_test_path = '/content/drive/MyDrive/DATA'

image_size = EfficientNet.get_image_size(model_name)
print(image_size)

model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
model = model.to(device)

dill.loads(dill.dumps(lambda x: x.rotate(90)))

transforms_train = transforms.Compose([
                                        transforms.Resize([int(600), int(600)], interpolation=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        #transforms.Lambda(lambda x: x.rotate(90)),
                                        transforms.RandomRotation(10),
                                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])

transforms_val = transforms.Compose([
                                        transforms.Resize([int(600), int(600)], interpolation=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])


train_data_set = datasets.ImageFolder(data_train_path, transform=transforms_train)
val_data_set = datasets.ImageFolder(data_validation_path, transform=transforms_val)

dataloaders, batch_num = {}, {}

dataloaders['train'] = DataLoader(train_data_set,
                                    batch_size=hyper_param_batch,
                                    shuffle=True,
                                    num_workers=0)
dataloaders['val'] = DataLoader(val_data_set,
                                    batch_size=hyper_param_batch,
                                    shuffle=False,
                                    num_workers=0)

batch_num['train'], batch_num['val'] = len(train_data_set), len(val_data_set)

print('batch_size : %d,  train/val : %d / %d' % (hyper_param_batch, batch_num['train'], batch_num['val']))

class_names = train_data_set.classes
print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                
            epoch_end = time.time() - epoch_start
            
            print('Training epochs {} in {:.0f}m {:.0f}s'.format(epoch, epoch_end // 60, epoch_end % 60))
            print()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))
    print('recall : ', recall_score(train_acc, val_acc))

    model.load_state_dict(best_model_wts)
        
    torch.save(model, PATH + 'aram_'+train_name+'.pt')
    torch.save(model.state_dict(), PATH + 'president_aram_'+train_name+'.pt')
    print('model saved')

    end_sec = time.time() - start_time
    end_times = str(datetime.timedelta(seconds=end_sec)).split('.')
    end_time = end_times[0]
    print("end time :", end_time)
    
    return model, best_idx, best_acc, train_loss, train_acc, val_loss, val_acc


criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model.parameters(),lr = 1e-4)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

num_epochs = 2
train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
