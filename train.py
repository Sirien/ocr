from __future__ import absolute_import, print_function, unicode_literals, division

import copy
import os
import os.path as osp
import sys

import glob
from skimage import io, transform
import pickle

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.models as models


sys.path.append('./')
from densenet import *

CHARSET_SIZE = 5611
INPUT_SIZE = 19
BATCH_SIZE = 32

LEARNING_RATE = 0.001
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

DICTIONARY = None
with open('./clabels_new_new', 'rb') as f:
    DICTIONARY = pickle.load(f)
# print(DICTIONARY)
# input()

class ChineseDataset(Dataset):
    def __init__(self, root='./dataset', transform=None, istrain=True):
        self.root_dir = root
        self.transform = transform
        self.istrain = istrain
        self.paths = []
        self.labels = []

        self.collect_paths()

    def collect_paths(self):
        base_dir = self.root_dir
        if self.istrain:
            base_dir = osp.join(base_dir, 'train')
        else:
            base_dir = osp.join(base_dir, 'test')
        for d in os.listdir(base_dir):
            label = int(d)
            char_dir = osp.join(base_dir, d)
            images = glob.glob("{}/*.png".format(char_dir))
            labels = [label for i in range(len(images))]
            
            self.paths += images
            self.labels += labels
        # print(min(self.labels), max(self.labels))
        # input()

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = io.imread(self.paths[idx])
        img = (255 - img) / 255.0
        img = np.expand_dims(img, axis=0)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_dataset(transform=None):
    train_dataset = ChineseDataset(transform=transform, istrain=True)
    test_dataset = ChineseDataset(transform=transform, istrain=False)
    return train_dataset, test_dataset

def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_model(model_name):
    if model_name == 'densenet':
        return DenseNet(num_init_features=1, num_classes=CHARSET_SIZE)
    elif model_name == 'resnet':
        model = resnet18(num_classes=CHARSET_SIZE)
        model.in_planes = 1
        return model
    else:
        # 定义自己的网络
        return NotImplementedError

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, is_inception=False, checkpoint_path='./Checkpoint'):
    # TODO: tensorboard
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if not osp.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    if osp.exists(osp.join(checkpoint_path, 'model.pth')):
        print("=====[Loading pretrained model]=====")
        model.load_state_dict(torch.load(osp.join(checkpoint_path, 'model.pth')))
    start_epoch = 0
    if osp.exists(osp.join(checkpoint_path, 'epoch.note')):
        with open(osp.join(checkpoint_path, 'epoch.note'), 'r') as f:
            start_epoch = int(f.readline().strip())
    for epoch in range(start_epoch, num_epochs):
        num_iter = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                num_iter += 1

                inputs = inputs.to(DEVICE).float()
                labels = labels.to(DEVICE)

                # print("inputs shape: ", inputs.shape, inputs.type())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        # print(outputs.shape)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_ = torch.sum(preds == labels.data)
                if num_iter % 50 == 0: 
                    print('Iter: {} Loss: {:.4f} Acc: {:.4f}'.format(num_iter, loss.item() * inputs.size(0), running_corrects_/float(BATCH_SIZE)))
                
                # save model
                if num_iter % 500 == 499:
                    torch.save(model.state_dict(), osp.join(checkpoint_path, 'model.pth'))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                with open(osp.join(checkpoint_path, 'acc.log'), 'w') as f:
                    for acc in val_acc_history:
                        f.write('epoch: {}, acc: {}\n'.format(epoch, acc))
                    f.flush()
            
            torch.save(model.state_dict(), osp.join(checkpoint_path, 'model_{}.pth'.format(epoch)))

            with open(osp.join(checkpoint_path, 'epoch.note'), 'w') as f:
                f.write(str(epoch))

           
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def inference(model, img_dir, label, is_inception=False):
    imgs = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
    imgs = [io.imread(img) for img in imgs]
    imgs = [(255 - img) / 255.0 for img in imgs]
    imgs = [np.expand_dims(img, axis=0) for img in imgs]
    imgs = [np.expand_dims(img, axis=0) for img in imgs]
    imgs = [torch.Tensor(img) for img in imgs]
    img_input = torch.cat(imgs, dim=0)

    model.eval()    
    with torch.no_grad():
        img_input = img_input.to(DEVICE)
        if is_inception and phase == 'train':
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

            outputs, aux_outputs = model(img)
        else:
            outputs = model(img_input)
        _, preds = torch.max(outputs, 1)
        #L = int(preds.cpu().numpy()[0])
        running_corrects_ = torch.sum(preds == label)
        #print("prediction: {} {}".format(DICTIONARY[L], L))
        print("acc of", DICTIONARY[label], float(running_corrects_)/img_input.shape[0])
        #return int(preds.cpu().numpy()[0]) == label
        return running_corrects_.cpu().numpy()

def train():
    train_dataset, test_dataset = get_dataset()
    train_dataloader, test_dataloader = get_dataloader(train_dataset), get_dataloader(test_dataset)
    dataloaders = {'train': train_dataloader, 'val': test_dataloader}

    model = get_model('densenet')
    model = model.to(DEVICE)
    # print("model===")
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_model, val_acc_history = train_model(model, dataloaders, criterion, optimizer)

    torch.save(model, osp.join(checkpoint_path, 'best_model.pth'))
    with open(osp.join(checkpoint_path, 'final_acc.log'), 'a') as f:
        for acc in val_acc_history:
            f.write('epoch: {}, acc: {}\n'.format(epoch, acc))
        f.flush()


def test_ocr():

    infer_result = dict()

    model = get_model('densenet')
    model = model.to(DEVICE)

    checkpoint_path="./Checkpoint"
    if osp.exists(osp.join(checkpoint_path, 'model.pth')):
        print("=====[Loading pretrained model]=====")
        model.load_state_dict(torch.load(osp.join(checkpoint_path, 'model_4.pth')))

    #dataset_dir = "./dataset/test"
    dataset_dir = "./cut_dataset/chars/"
    for _idx_d,d in enumerate(os.listdir(dataset_dir)):

        print(_idx_d)

        if (len(os.listdir(os.path.join(dataset_dir,d))) == 0):
            # print("no image in this folder")
            continue

        label = int(d)

        count = len(os.listdir(os.path.join(dataset_dir,d)))
        T_count = inference(model, os.path.join(dataset_dir, d), label)

        infer_result[DICTIONARY[label]] = [T_count, count]

    pickle.dump(infer_result, open("infer_result4.pkl", "wb"))


if __name__ == '__main__':
    test_ocr()
    # train()
