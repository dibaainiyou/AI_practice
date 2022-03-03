import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        Dataset.__init__(self)
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.leng=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]

    def __len__(self):
        return self.leng

dataset=DiabetesDataset("diabetes.csv")
train_loader=DataLoader(dataset=dataset,batch_size=3,shuffle=True,num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        y_pre=self.sigmoid(self.linear2(x))
        return y_pre



model=Model()
criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


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



    x_test=torch.tensor([[-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333],
                         [-0.882353,-0.145729,0.0819672,-0.414141,0,-0.207153,-0.766866,-0.666667]])                              #测试例子
    y_test=model(x_test)
    print(y_test)







