import torch
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

#googlenet的inception
class inception(torch.nn.Module):
    def __init__(self,in_channal=1):
        torch.nn.Module.__init__(self)
        self.avg=torch.nn.AvgPool2d(stride=1,padding=1,kernel_size=3)                   #第一个分支是一个avg池化加1x1的卷积层
        self.conv0=torch.nn.Conv2d(1,10,kernel_size=1)

        self.conv1=torch.nn.Conv2d(1,10,kernel_size=1)                                  #第二个分支是一个1x1的卷积

        self.conv2=torch.nn.Conv2d(1,10,kernel_size=1)                                  #第三个分支是1x1的卷积加5x5的卷积
        self.conv22 = torch.nn.Conv2d(10, 10, kernel_size=5,padding=2)

        self.conv3= torch.nn.Conv2d(1, 10, kernel_size=1)                                #第四个分支是1x1和两个3x3的卷积
        self.conv32= torch.nn.Conv2d(10, 10, kernel_size=3,padding=1)
        self.conv33= torch.nn.Conv2d(10, 10, kernel_size=3,padding=1)

    def forward(self,x):
        branch1=self.conv0(self.avg(x))
        branch2=self.conv1(x)
        branch3=self.conv22(self.conv2(x))
        branch4=self.conv33(self.conv32(self.conv3(x)))
        outputs=[branch1,branch2,branch3,branch4]
        a=torch.cat(outputs, dim=1)
        return a


class Net(torch.nn.Module):                                                         #整个网络先进入inception再到3x3的卷积层再到4x4的最大池化层后展开为360的全连接层
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.inception=inception(1)
        self.L2=torch.nn.Conv2d(40,10,kernel_size=3)
        self.L3=torch.nn.MaxPool2d(kernel_size=4,stride=4)
        self.Fc=torch.nn.Linear(360,10)


    def forward(self,x):
        number=x.size(0)
        x=self.inception(x)
        x=self.L2(x)
        x=self.L3(x)
        x=x.view(number,-1)
        x=self.Fc(x)
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
