import torch
from torch import nn
import time
import numpy as np
import random
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset




 
 





#初始化设置
BATCH_SIZE = 16
num_epochs = 100


transform = transforms.Compose([

    transforms.Normalize(mean=[0.485,  ],std=[0.229,  ]) #正则化
])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=512):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

net = ResNet50().to('cuda') 

with open('./pos_weight.txt', 'r') as hh:
    weight = hh.readlines()
pos_weight=weight[0].split()
pos_weight=list(map(float,pos_weight))
pos_weight=torch.tensor(pos_weight).cuda()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001,betas=(0.9,0.99)) 
pos_weight=pos_weight

def one_hot(x, class_count):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]




with open('./zz/train/train.txt', 'r') as f:
    train = f.readlines()
with open('./zz/test/test.txt', 'r') as g:
    test = g.readlines()

train_data = []
train_label = []
print(len(train))
# 读取数据
for i in range(len(train)):
    labels = []
    label=train[i].split()
    label=list(map(float,label))
    img_path='./zz/train/'+str(i)+'.txt'
    img = np.loadtxt(img_path,delimiter=' ')
    img=torch.tensor(img)
    img=img.reshape(1,32,32)
    img=transform(img)
    train_data.append(img.float())
    labels.append(label)
    train_label.append(torch.tensor(labels))


test_data = []
test_label = []

for i in range(len(test)):
    labels = []
    label=test[i].split()
    label=list(map(float,label))
    img_path='./zz/test/'+str(i)+'.txt'
    img = np.loadtxt(img_path,delimiter=' ')
    img=torch.tensor(img)
    #print(img.shape)
    img=img.reshape(1,32,32)
    img=transform(img)
    test_data.append(img.float())
    labels.append(label)
    test_label.append(torch.tensor(labels))

#test_label=torch.tensor(test_label);
#train_label=torch.tensor(train_label);
test_label= torch.tensor([item.cpu().detach().numpy() for item in test_label]).cuda()
train_label= torch.tensor([item.cpu().detach().numpy() for item in train_label]).cuda()
print(test_label.shape)
test_label=torch.squeeze(test_label)
train_label=torch.squeeze(train_label)
test_label=torch.cuda.FloatTensor(test_label)
train_label=torch.cuda.FloatTensor(train_label)
train_data_size = len(train)
valid_data_size = len(test)
#test_label=test_label.reshape(valid_data_size,1)
#train_label=train_label.reshape(test_data_size,1)
#test_label=one_hot(test_label,44)
#train_label=one_hot(train_label,44)



print(train_data_size,valid_data_size)

class GetData(Dataset):

    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, data, label):

        # self
        self.data = data
        self.label = label
       

    def __getitem__(self, idx):
        
        img = self.data[idx];
        label = self.label[idx];
        return img, label

    def __len__(self):
        return len(self.label)



train_dataset = GetData(train_data, train_label)
test_dataset = GetData(test_data, test_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #加载数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0) #加载数据


def train_and_valid(model, loss_function, optimizer, epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
    best_p_epoch=0
    best_p_acc=0
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_p_acc=0
        valid_p_acc=0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
 
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
 
            loss.backward()
 
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
 
            #ret, predictions = torch.max(outputs.data, 1)
            #correct_counts = predictions.eq(labels.data.view_as(predictions))
            #print(predictions)
            #print(labels)
            #input()
            sh=outputs.data.shape
            temp=0
            point_acc=0
            pred_label_locate = torch.argsort(outputs, descending=True)[:, 0:25].cuda()
            for i in range(outputs.shape[0]):
                temp_label = torch.zeros(1, outputs.shape[1]).cuda()
                temp_label[0,pred_label_locate[i]] = 1
                target_one_num = torch.sum(labels[i])
                true_predict_num = torch.sum(torch.dot(temp_label.squeeze(),labels[i]))
                correction=temp_label.squeeze()-labels[i]
                correction=correction.cpu()
                corr=np.sqrt(torch.dot(correction,correction))
        # 对每一幅图像进行预测准确率的计算
                train_acc += (1-corr / 512)
        # 对每一幅图像进行预测查全率的计算
                train_p_acc+= true_predict_num / target_one_num
           
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
               
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)
 
                temp=0
                point_acc=0
                sh=outputs.data.shape
                pred_label_locate = torch.argsort(outputs, descending=True)[:, 0:25].cuda()
                for i in range(outputs.shape[0]):
                    temp_label = torch.zeros(1, outputs.shape[1]).cuda()
                    temp_label[0,pred_label_locate[i]] = 1
                    target_one_num = torch.sum(labels[i])
                    true_predict_num = torch.sum(torch.dot(temp_label.squeeze(),labels[i]))
                    correction=temp_label.squeeze()-labels[i]
                    correction=correction.cpu()
                    corr=np.sqrt(torch.dot(correction,correction))
                    #print(corr)
        # 对每一幅图像进行预测准确率的计算
                    valid_acc += (1-corr / 512)
        # 对每一幅图像进行预测查全率的计算
                    valid_p_acc+= true_predict_num / target_one_num

 

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
        avg_tp_acc = train_p_acc/train_data_size
        avg_vp_acc = valid_p_acc/valid_data_size
        avg_valid_loss = valid_loss/valid_data_size
        
        avg_valid_acc = valid_acc/valid_data_size
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
        if  best_p_acc<avg_vp_acc:
            best_p_acc = avg_vp_acc
            best_p_epoch = epoch + 1
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%,PointAccuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, PointAccuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100,avg_tp_acc*100,avg_valid_loss, avg_valid_acc*100,avg_vp_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        print("Best Accuracy for Point : {:.4f} at epoch {:03d}".format(best_p_acc, best_p_epoch))
    return history


history = train_and_valid(net, loss_function, optimizer, num_epochs)

#将训练参数用图表示出来
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1.1)

plt.show()
 
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)

plt.show()








































