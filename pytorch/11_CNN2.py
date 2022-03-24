import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])       #对minist进行正则化
#训练和测试训练集
train_dataset=datasets.MNIST(root='../dataset/mnist/',
                          train=True,
                          download=True,
                          transform=transform
)
train_loader=DataLoader(train_dataset,
                          shuffle=True,
                        batch_size=batch_size
)
test_dataset=datasets.MNIST(root='../dataset/mnist/',
                          train=False,
                          download=True,
                          transform=transform
)
test_loader=DataLoader(test_dataset,
                          shuffle=False,
                        batch_size=batch_size
)


class ResidualBlock(torch.nn.Module):
    def __init__(self,channels=1):
        torch.nn.Module.__init__(self)
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        y=self.relu(self.conv1(x))
        y=self.relu(self.conv2(y))
        return F.relu(x+y)

#先一个卷积再一个残缺再一个卷积再一个残缺
class Net(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp=torch.nn.MaxPool2d(2)
        self.rblock1=ResidualBlock(16)
        self.rblock2=ResidualBlock(32)
        self.fc=torch.nn.Linear(512,10)
        self.relu=torch.nn.ReLU()


    def forward(self,x):
        number=x.size(0)
        x=self.mp(self.relu(self.conv1(x)))
        x=self.rblock1(x)
        x=self.mp(self.relu(self.conv2(x)))
        x = self.rblock2(x)
        x=x.view(number,-1)
        x=self.fc(x)
        return x

model=Net()
device=torch.device('cuda:0')                                      #到gpu运算
model=model.to(device)
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):                                                   #训练
    running_loss=0.0
    for batch_idex,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
#        inputs=inputs.view(-1,1,28,28)

        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idex%300==299:
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idex+1,running_loss/300))
            running_loss=0.0

def test():                                                               #测试
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            everyline_max,predicted=torch.max(outputs,dim=1)         #网络结果是多行多列的矩阵，利用这个可以找出最大值和最大值的下标（即分类）
            total +=labels.size(0)
            correct +=(predicted==labels).sum().item()
        print('Arruracy on test set:%d %%'%(100*correct/total))


if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
