
import cv2
import random
from tqdm import tqdm
from model import Vgg16Net

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.models as models


Epochs = 2000
learning_rate = 0.001

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Vgg16Net().to(device)
model.train()
summary(model, (3, 32, 32))
loss_func = nn.CrossEntropyLoss()

with open('./train_label.txt', 'r') as f:
    train = f.readlines()
with open('./test_label.txt', 'r') as g:
    test = g.readlines()
def test_ACC(img, label, Accuracy):
    out = model(img)
    _, pre_tensor = torch.max(out, 1)

    if pre_tensor == label:
        Accuracy.append(1)
    else:
        Accuracy.append(0)

train_data = []
train_lable = []

# 读取数据
for i in range(len(train)):
    labels = []
    label=float(train[i])
    img_path='./figures/'+str(i+1)+'.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = torch.from_numpy(img).to(device).div(255.0).unsqueeze(0) 
    img = img.permute(0, 3, 1, 2) 
    train_data.append(img.float())
    labels.append(int(label))
    train_lable.append(torch.tensor(labels).to(device))


test_data = []
test_lable = []

for i in range(len(test)):
    img_path='./test/'+str(i+1)+'.png'
    labels = []
    label=float(test[i])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = torch.from_numpy(img).to(device).div(255.0).unsqueeze(0) 
    img = img.permute(0, 3, 1, 2)  
    test_data.append(img.float())
    labels.append(int(label))
    test_lable.append(torch.tensor(labels).to(device))    


# 开始训练
for i in range(1, Epochs+1):
    batch = 1
    ACC_Loss = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for f in tqdm(range(0, 256), desc='Epoch: ' + str(i), postfix={'lr':learning_rate}, colour='red'):
        rand_num = random.randint(0, len(train)-1)
        out = model(train_data[rand_num])
        loss = loss_func(out, train_lable[rand_num])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ACC_Loss = (ACC_Loss + loss.item()) / batch
        batch += 1

    log_train = 'Train Epoch: ' + str(i) + ', Learning_rate: ' + str(learning_rate) + ', Batch: ' + str(batch - 1) + ', ACC_LOSS: ' + str(ACC_Loss)
    print(log_train)

    if i % 10 == 0:
        model.eval()
        Accuracy = []
        for num_b in tqdm(range(0, len(test_data)), desc='Test: '):
            test_ACC(test_data[num_b], test_lable[num_b], Accuracy)
            num = 0
            for j in Accuracy:
                num += j
        test = 'Test: Data Volume: {}, Accuracy: {:.2f} %'.format(len(Accuracy), num / len(Accuracy) * 100)
        print(test + '\n')
        with open('log.txt', 'a') as f:
                f.write(log_train + '\n' + test + '\n')

    if i % 10 == 0:
        torch.save(model.state_dict(), './VGG/' + str(i) + '.pt')

