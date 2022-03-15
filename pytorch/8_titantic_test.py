import csv
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

text=list(csv.reader(open('titantic/test.csv',encoding='utf-8')))
text.remove(text[0])
text_afterchoose=[]                                     #对特征进行选取

               #准备对数据进行正则化

avg=29.66
fare_avg=32.2042079685746
sibsp_avg=0.5230078563411896
parch_avg=0.38159371492704824


for i in range(418):                                     #创建一个891行的列表来接收选取特征的列表
    text_afterchoose.append([])

for i in range(418):
    if text[i][3] == 'male':                              #male设为1
        text[i][3] = 1
    else:
        text[i][3]=0

    if text[i][4]=='':                                      #age没有取平均值
        text[i][4]=avg
    if text[i][8]=='':
        text[i][8]=fare_avg

    text[i][4]=float(text[i][4])/avg
    text[i][5] = float(text[i][5]) /sibsp_avg
    text[i][6] = float(text[i][6]) / parch_avg
    text[i][8]=float(text[i][8])/fare_avg

    for j in (3,4,5,6,8):
        text_afterchoose[i].append(float(text[i][j]))


text_afterchoose=np.array(text_afterchoose,dtype='float32')




class titanticDataset(Dataset):                                    #数据集类的继承
     def __init__(self):
         Dataset.__init__(self)
         xy=text_afterchoose      #将数据读入
         self.leng=xy.shape[0]
         self.x_data=torch.from_numpy(xy)


     def __getitem__(self, item):
         return self.x_data[item],self.y_data[item]

     def __len__(self):
         return self.leng


dataset=titanticDataset()


class Model(torch.nn.Module):  # 三层隐藏层的网络
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.linear1 = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)
        self.linear4 = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        y_pre = self.sigmoid(self.linear4(x))
        return y_pre


model = Model()
model.load_state_dict(torch.load('state_dict.pkl'))

y_test=model(dataset.x_data)
right=0
for i in range(418):
    if y_test[i].item()>0.5:
        y_test[i]=1
    else:
        y_test[i]=0

with open('titantic_pre.csv','w',newline='') as f:
    f_csv = csv.writer(f)
    for i in range(418):
        f_csv.writerow(str(int(y_test[i].item())))

