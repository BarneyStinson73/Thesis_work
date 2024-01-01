import numpy as np
# computing every step manually

#assumption :linear regression
# f=w*x

#actual fuction f=2*x

X=np.array([1,2,3,4],dtype=np.float32)
Y=np.array([2,4,6,8],dtype=np.float32)
w=0.0

# forward pass,then gradient calculation,then backward pass

def forward(x):
    return w*x

# for loss we are taking MSE(Mean of Squares)
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# J=1/N * (w*x-y)**2
# dJ/dw=1/N 2x(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5)={forward(5):.3f}')

# Training
learning_rate=0.01
n_iters=20

for epoch in range(n_iters):
    y_pred=forward(X)
    l=loss(Y,y_pred)
    dw=gradient(X,Y,y_pred)
    w=w-learning_rate*dw
    if epoch%2==0:
        print(f'epoch {epoch+1}: w={w:.3f},loss={l:.8f}')
        
print(f'Prediction after training: f(5)={forward(5):.3f}')