import numpy
import torch

x_list=[1.0,2.0,3.0]
y_list=[2.0,4.0,6.0]

w1=torch.tensor([1.0])
w1.requires_grad=True
w2=torch.tensor([1.0])
w2.requires_grad=True
b=torch.tensor([1.0])
b.requires_grad=True

def forward(x):
    return w1*x*x+w2*x+b

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2


print("predict(before training)",1,forward(1).item())

for epoch in range(1000):
    for x,y in zip(x_list,y_list):
        l=loss(x,y)
        l.backward()
        print("\tgrad:",x,y,w1.item())
        w1.data=w1.data-0.01*w1.grad.data
        w1.grad.data.zero_()

        w2.data = w2.data - 0.01 * w2.grad.data
        w2.grad.data.zero_()

        b.data = b.data - 0.01 * b.grad.data
        b.grad.data.zero_()
    print("progress:",epoch,l.item())
print("predict:",1,forward(1).item())
print("predict:",2,forward(2).item())
print("predict:",3,forward(3).item())

