import torch
import numpy as np

# for gradient calculation ,has to put requires_grad= True
x= torch.randn(3,requires_grad=True)
print(x)
y=x+2
print(y)
z=y*y*2
print(z)
#calculating gradient with scalar value
# z=z.mean()
# print(z)
# z.backward()
# print(x.grad)

# gradient with a vector rather than a scalar
# v=torch.tensor([0.1,1.0,0.001],dtype=torch.float)
# z.backward(v)
# print(x.grad)

# turing gradient = True to False
# 3 process -->
# 1st:
# x.requires_grad_(False);
# print(x)
#2nd:
# y=x.detach()
# print(y)
#3rd:
# with torch.no_grad():
#     y=x+2
#     print(y)

#optimization
weights= torch.ones(4,requires_grad=True)
optimizer=torch.optim.SGD(weights,lr=0.01)
optimizer.step()
optimizer.zero_grad() # necessary operation in every iteration if we have to do it in loop,otherwise previous values will keep accumulating