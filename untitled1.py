# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 01:02:51 2019

@author: 米
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:46:33 2019

@author: 米
"""

import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from skimage import io
import os
import scipy.misc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif']=['SimHei']
#用以正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False
#用正常显示负号
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
 
# Training settings

batch_size = 20
#每一批数据的个数
num_workers=0
#处理器个数0
valid_size=0.2
#验证集比例0.2
# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
 
test_dataset = datasets.MNIST(root='./data/',
               train=False,
               transform=transforms.ToTensor())
#转化为张量模式

num_train =len(train_dataset)
indices=list(range(num_train))
np.random.shuffle(indices)
split = int (np.floor(valid_size*num_train))
train_idx,valid_idx =indices[split:],indices[:split]
#随机打乱取1/5作为测试集

#划分新的测试集，训练集
train_sampler=SubsetRandomSampler(train_idx)
valid_sampler=SubsetRandomSampler(valid_idx)


# Data Loader (Input Pipeline)
#pytorch 数据集需要通过Dataloader载入
#其实质是python中的生成器 ，每一次调用返回一个batch的数据
"""train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=batch_size,
                     shuffle=False)"""
train_loader = torch.utils.data.DataLoader(train_dataset,
                      batch_size=batch_size,
                      sampler=train_sampler,
                      num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_dataset,
                      batch_size=batch_size,
                      sampler=valid_sampler,
                      num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset,
                      batch_size=batch_size,
                      num_workers=num_workers)
 #创建训练集、验证集、测试集的数据loader
'''
dataiter = iter (train_loader)
images,labels=dataiter.next()
images=images.numpy()

 #绘图
fig =plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title(str(labels[idx].item()))
plt.show
 '''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    hidden_1=512
    hidden_2=512
    # 输入1通道，输出10通道，kernel 5*5
   # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #self.mp = nn.MaxPool2d(2)
    
    #self.fc = nn.Linear(320, 10)
    self.fc1 = nn.Linear(28*28,hidden_1)
    
    self.fc2 = nn.Linear(hidden_1,hidden_2)
    #512 到 512
    #self.mp = nn.MaxPool2d(2)

    self.fc3 = nn.Linear(hidden_2,10)
    #512->10
    self.dropout = nn.Dropout(0.2)
    #防止过拟合
    
  def forward(self, x):
      x=x.view(-1,28*28)
      #拉成784的长向量
      x=F.relu(self.fc1(x))
      #第一层，激活函数relu
      x=self.dropout(x)
      
      x=F.relu(self.fc2(x))
      #第二层，激活函数relu
      x=self.dropout(x)

      #x = F.relu(self.mp(self.conv1(x)))
      
      x=self.fc3(x)
      return x
    # in_size = 64
 #   in_size = x.size(0) # one batch
    # x: 64*10*12*12
 #   x = F.relu(self.mp(self.conv1(x)))
    # x: 64*20*4*4
 #   x = F.relu(self.mp(self.conv2(x)))
    # x: 64*320
  #  x = x.view(in_size, -1) # flatten the tensor
    # x: 64*10
  #  x = self.fc(x)
  #  return F.log_softmax(x)
 
 
#实例化模型 
model = Net()
print(model)
 
#criterion=F.cross_entropy()
criterion=nn.crossentropy()
#损失函数为交叉熵损失函数
#criterion=fun_self,看规定格式

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#随机梯度下降sgd优化器，学习率0.01,
 
jsq=0
drop=0.01#初始学习率


len(train_loader.dataset)

n_epochs=100
#100 轮，连续五轮没有下降，则学习率除以10

#训练轮数，每一轮都遍历所有图像
#初始化误差为正无穷
valid_loss_min = np.inf


#用列表存储训练损失和验证损失
train_loss_list = []
val_loss_list=[]
weight0 = weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = weight9 = 1.0


#每一轮
for epoch in range(n_epochs):
    if jsq==5:
        drop=drop*0.9
        print('验证损失连续五轮无变化，学习率降低10%,进入下一轮训练')
        jsq=0
        
    optimizer = torch.optim.SGD(model.parameters(),lr=drop,momentum=0.5)
    #初始化训练损失和验证损失
    train_loss=0.0
    valid_loss=0.0
    
    #训练阶段
    model.train()#调整为训练模式
    #获取一批次数据与标签
    for data,target in train_loader:
        #所有梯度归零
        optimizer.zero_grad()
        #正向推断，预测结果
        output = model(data)
    #预测结果和标签进行比较，求cross entrorpy    
        loss = criterion(output,target)
    #反向传播，求梯度
        loss.backward()
        #优化(权重更新)
        optimizer.step()
        #将本批次所有样本的损失函数值求和，作为训练损失
        train_loss += loss.item()*data.size(0)

    #验证阶段
    model.eval()
    
    count0=count1=count2=count3=count4=count5=count6=count7=count8=count9=0
    correct0=correct1=correct2=correct3=correct4=correct5=correct6=correct7=correct8=correct9=0
    #weight0 = weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = weight9 = 1.0
    m=0
    #验证阶段，关闭dropout和Bn层
    #获得一个批次的数据及标签
    for data,target in valid_loader:
        output = model(data)#正向预测分类
        loss =criterion(output,target)
        #计算交叉熵
        valid_loss+=loss.item()*data.size(0)
    #求和作为验证值
    
#结束该论，打印训练和验证指标
#列表存储平均验证和平均训练损失
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_loss_list.append(train_loss)
    val_loss_list.append(valid_loss)

    jsq+=1
    print('第{}轮 \t 训练损失：{:.6f} \t验证损失：{:.6f}'.format(epoch+1,train_loss,valid_loss))
    
    
    if valid_loss <= valid_loss_min:
        print('验证损失相比之前降低了({:.6f} -->{:.6f}).保存模型'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(),'model.pt')
        valid_loss_min =valid_loss  
        jsq = 0
        
    
plt.plot(train_loss_list,label='训练误差')
plt.plot(val_loss_list,label='验证误差')
plt.legend()
plt.title('误差变化')
plt.show()
    
plt.plot(val_loss_list,label='验证误差',c='r')
plt.title('验证集误差变化')
plt.show()

model.load_state_dict(torch.load('model.pt'))

number=0
times=0
#初始化误差
test_loss=0.0
class_correct= list(0.for i in range(10))
#识别个数
class_total= list(0.for i in range(10))
#总样本
erorr_five_list=[]


model.eval()
for data,target in test_loader:
    output=model(data)
    loss=criterion(output,target)
    test_loss += loss.item()*data.size(0)
    #最大值作为预测分类
    _,pred = torch.max(output,1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))#置为0，1
    for i in range(batch_size):
        label = target.data[i]#LABEL为正确标签
        class_correct[label]+=correct[i].item()
        #实际分类对则加1，错则加0
        class_total[label]+=1
        #正确类加一
     '''   if label==5 and correct[i].item()==0:
            print('5')'''
            
        
#计算平均测试误差
test_loss = test_loss/len(test_loader.dataset)
print('测试集上的误差:{:.6f}'.format(test_loss))
for i in range(10):
    print('数字{}在测试集上的识别正确率为{:.5f}% ({:.0f}/{:.0f})'.format(i,class_correct[i]*100/class_total[i],class_correct[i],class_total[i]))


dataiter = iter(test_loader)
images,labels =dataiter.next()

output = model(images)

_,preds =torch.max(output,1)
#取概率最高的作为分类结果
images =images.numpy()

#可视化

fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
   #idx in np.arange(20):
    ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()),str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))
plt.show

'''image = np.asarray(data[2]).squeeze()
plt.imshow(image)
plt.show()
def fun_self( output, target):
   outpu
   return'''