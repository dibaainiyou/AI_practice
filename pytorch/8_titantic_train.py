import csv

import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

text=list(csv.reader(open('titantic/train.csv',encoding='utf-8')))
text.remove(text[0])
text_afterchoose=[]                                     #对特征进行选取

sum,fare_sum,sibsp_sum,parch_sum=0,0,0,0                #准备对数据进行正则化
for i in range(891):                                    #对年龄缺失的处理
    if text[i][5]!='':
        sum+=float(text[i][5])
avg=sum/715

for i in range(891):                                    #求票价的平均值
    fare_sum+=float(text[i][9])
    sibsp_sum+=float(text[i][6])
    parch_sum += float(text[i][7])
fare_avg=fare_sum/891
sibsp_avg=sibsp_sum/891
parch_avg=parch_sum/891


for i in range(891):                                     #创建一个891行的列表来接收选取特征的列表
    text_afterchoose.append([])

for i in range(891):
    if text[i][4] == 'male':                              #male设为1
        text[i][4] = 1
    else:
        text[i][4]=0

    if text[i][5]=='':                                      #age没有取平均值
        text[i][5]=avg

    text[i][5]=float(text[i][5])/avg
    text[i][6] = float(text[i][6]) /sibsp_avg
    text[i][7] = float(text[i][7]) / parch_avg
    text[i][9]=float(text[i][9])/fare_avg

    for j in (1,4,5,6,7,9):
        text_afterchoose[i].append(float(text[i][j]))


text_afterchoose=np.array(text_afterchoose,dtype='float32')




class titanticDataset(Dataset):                                    #数据集类的继承
     def __init__(self):
         Dataset.__init__(self)
         xy=text_afterchoose      #将数据读入
         self.leng=xy.shape[0]
         self.x_data=torch.from_numpy(xy[:,1:])                     #数据分为特征和标签
         self.y_data=torch.from_numpy(xy[:,[0]])

     def __getitem__(self, item):
         return self.x_data[item],self.y_data[item]

     def __len__(self):
         return self.leng

dataset=titanticDataset()
train_loader=DataLoader(dataset=dataset,batch_size=3,shuffle=True,num_workers=0)       #mini_batch

class Model(torch.nn.Module):                                                          #一层隐藏层的网络
     def __init__(self):
         torch.nn.Module.__init__(self)
         self.linear1=torch.nn.Linear(5,10)
         self.linear2=torch.nn.Linear(10,10)
         self.linear3 = torch.nn.Linear(10, 10)
         self.linear4=torch.nn.Linear(10,1)
         self.sigmoid=torch.nn.Sigmoid()

     def forward(self,x):
         x=self.sigmoid(self.linear1(x))
         x=self.sigmoid(self.linear2(x))
         x = self.sigmoid(self.linear3(x))
         y_pre=self.sigmoid(self.linear4(x))
         return y_pre



model=Model()
criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)


if __name__ == '__main__':
     for epoch in range(100):                                 #进行1000次计算更新
         for i,data in enumerate(train_loader,0):
             inputs,labels=data
             y_pred=model.forward(inputs)
             loss=criterion(y_pred,labels)
             print(epoch,i,loss.item())
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()


                            #测试例子
y_test=model(dataset.x_data)
right=0
for i in range(891):
    if y_test[i].item()>0.5:
        y_test[i]=1
    else:
        y_test[i]=0
    if dataset.y_data[i]==y_test[i].item():
        right+=1

print(right)

#训练一百轮，大约83%准确度，测试集还未编写







