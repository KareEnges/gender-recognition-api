import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# import os
# import torch.distributed as dist

# os.environ['GLOO_SOCKET_IFNAME'] = 'WLAN'
# dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=2)


def loadtraindata():
    path = r"./train"
    # path = r"./train2"
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((300, 300)),  # 将图片缩放到指定大小（h,w）
                                                    transforms.CenterCrop(300),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 9)  # 卷积层
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 10, 11)  # 卷积层
        self.conv3 = nn.Conv2d(10, 6, 9)  # 卷积层
        self.conv4 = nn.Conv2d(6, 16, 11)  # 卷积层
        self.fc1 = nn.Linear(1600, 480)  # 全连接层
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 1600)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


classes = ('未知', '女', '男')



def trainandsave():  # 训练
    writer = SummaryWriter('runs/ANSWER')
    trainloader = loadtraindata()
    # trainloader = torch.utils.data.distributed.DistributedSampler(trainloader)
    net = Net()
    # net = nn.parallel.DistributedDataParallel(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(200):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            img_grid = torchvision.utils.make_grid(inputs)
            writer.add_image('Man', img_grid)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:
                writer.add_scalar('training loss', running_loss / 20, epoch * len(trainloader) + i)
                #writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels),
                 #                 global_step=epoch * len(trainloader) + i)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        print(str(epoch) + '  OK!')
        torch.save(net, './out/' + str(epoch) + 'net.pkl')
        torch.save(net.state_dict(), './out/' + str(epoch) + 'net_params.pkl')
    writer.add_graph(Net(), inputs)
    writer.close()
    print('Finished Training')


def reload_net():
    trainednet = torch.load('./net.pkl')
    return trainednet


def test_one(img_path):
    model = reload_net()
    transform_valid = transforms.Compose([
        transforms.Resize((300, 300), interpolation=4),
        transforms.ToTensor()
    ]
    )
    img = Image.open(img_path)
    img_ = transform_valid(img).unsqueeze(0)
    outputs = model(img_)
    _, indices = torch.max(outputs, 1)
    # print(indices)
    result = classes[indices]
    print('我认为这个东西的性别是:', result)
    print(outputs)
    return result


def main(url):
    return test_one(url)
    #trainandsave()
