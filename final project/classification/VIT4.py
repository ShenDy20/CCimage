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
num_epochs = 20


transform = transforms.Compose([

    transforms.Normalize(mean=[0.485,  ],std=[0.229,  ]) #正则化
])






# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_height,image_width,num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        patch_height=8
        patch_width = 8

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def VIT():
    return ViT(dim=128,
    image_height=64,
    image_width=32,
    num_classes=512,
    channels=1,
    depth=8,heads=32,mlp_dim=64)
net = VIT().to('cuda') 

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01,betas=(0.9,0.99)) 

def one_hot(x, class_count):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]




with open('./zzmore/train/train.txt', 'r') as f:
    train = f.readlines()
with open('./zzmore/test/test.txt', 'r') as g:
    test = g.readlines()

train_data = []
train_label = []
print(len(train))
# 读取数据
for i in range(len(train)):
    labels = []
    label=train[i].split()
    label=list(map(float,label))
    img_path='./zzmore/train/'+str(i)+'.txt'
    img = np.loadtxt(img_path,delimiter=' ')
    img=torch.tensor(img)

    img=img.reshape(1,64,32)
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
    img_path='./zzmore/test/'+str(i)+'.txt'
    img = np.loadtxt(img_path,delimiter=' ')
    img=torch.tensor(img)
    #print(img.shape)
    img=img.reshape(1,64,32)
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
                #print(correction)
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

torch.save(net, 'vit4.pt')
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
