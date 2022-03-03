
import torch

x_list=torch.Tensor([[1.0],[2.0],[3.0]])               #数据
y_list=torch.Tensor([[0],[0],[1]])
class Logisticmodel(torch.nn.Module):                    #线性模型类
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.linear=torch.nn.Linear(1,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        y_pred=self.sigmoid(self.linear(x))
        return y_pred



model=Logisticmodel()
criterion=torch.nn.BCELoss(reduction='sum')
optimzier=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):                                 #进行1000次计算更新
    y_pred=model.forward(x_list)
    loss=criterion(y_pred,y_list)
    print(epoch,loss)

    optimzier.zero_grad()
    loss.backward()
    optimzier.step()
    print('w=', model.linear.weight.item())



x_test=torch.tensor([[3.0]])                              #测试例子
y_test=model(x_test)
print(y_test)





