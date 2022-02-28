import torch

x_list=torch.tensor([[1.0],[2.0],[3.0]])               #数据
y_list=torch.tensor([[2.0],[4.0],[6.0]])
class Linearmodel(torch.nn.Module):                    #线性模型类
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred



model=Linearmodel()
criterion=torch.nn.MSELoss(reduction='sum')
optimzier=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):                                 #进行1000次计算更新
    y_pred=model(x_list)
    loss=criterion(y_list,y_pred)
    print(epoch,loss)

    optimzier.zero_grad()
    loss.backward()
    optimzier.step()
    print('w=', model.linear.weight.item())



x_test=torch.tensor([[4.0]])                              #测试例子
y_test=model(x_test)
print(y_test.data)





