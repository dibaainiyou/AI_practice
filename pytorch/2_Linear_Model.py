import numpy as np
import matplotlib.pyplot as plt

x_list=np.array([1,2,3])
y_list=np.array([2,4,6])
def oneloss(x,y,w,b):
    predict=w*x+b
    return (predict-y)*(predict-y)

w_list=np.arange(0,4.1,0.1)             #一个数值对的损失函数
b_list=np.arange(-2,2.1,0.1)
loss_list=[]

minLoss=1000
minwb=[]
for w in w_list:                         #循环对w,b进行损失函数计算
    for b in b_list:
        loss=0
        for x,y in zip(x_list,y_list):
            loss+=oneloss(x,y,w,b)
        if loss<minLoss:
            minLoss=loss
            minwb=[w,b]
        loss_list.append(loss/len(x_list))

print(minwb)                              #最小损失函数值的w，b
print(minLoss)                            #最小损失函数


loss_list=np.array(loss_list)
loss_list.resize((len(w_list),len(b_list)))
ww,bb=np.meshgrid(w_list,b_list)

plt.figure()                                #画三维图
lossPic=plt.axes(xlabel='w',ylabel='b',zlabel='loss',projection='3d')
lossPic.plot_surface(ww,bb,loss_list,cmap='rainbow')
plt.show()






