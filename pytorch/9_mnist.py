import torch
from torchvision import transforms
from torchvision import datasets                 #之前的Dataset改变
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

train_dataset=datasets.MNIST(root='../dataset/mnist/',
                          train=True,
                          download=True,
                          transform=transform
)
train_loader=DataLoader(train_dataset,
                          shuffle=True,
                        batch_size=batch_size,
                        num_workers=4
)
test_dataset=datasets.MNIST(root='../dataset/mnist/',
                          train=False,
                          download=True,
                          transform=transform
)
test_loader=DataLoader(test_dataset,
                          shuffle=False,
                        batch_size=batch_size,
                         num_workers=4
)


class Net(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.linear1=torch.nn.Linear(784,512)
        self.linear2=torch.nn.Linear(512,256)
        self.linear3=torch.nn.Linear(256,128)
        self.linear4=torch.nn.Linear(128, 64)
        self.linear5= torch.nn.Linear(64, 10)

    def forward(self,x):
        x=x.view(-1,784)
        x=self.linear1(F.relu(x))
        x=self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        return self.linear5(x)

model=Net()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idex,data in enumerate(train_loader,0):
        inputs,labels=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idex%300==299:
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idex+1,running_loss/300))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            everyline_max,predicted=torch.max(outputs,dim=1)         #网络结果是多行多列的矩阵，利用这个可以找出最大值和最大值的下标（即分类）
            total +=labels.size(0)
            correct +=(predicted==labels).sum().item()
        print('Arruracy on test set:%d %%'%(100*correct/total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
