import torch
import numpy as np
# x=torch.rand(3,5)
# print(x)
# y=x.view(-1,15)
# print(y)


# x=np.array([[1,2,3],[4,5,6]])
# print(x)
# y=torch.from_numpy(x)
# print(y.type())


# turn tensor to numpy and show the type
x=torch.rand(3,5)
print(x)
y=x.numpy()
print(y.dtype)