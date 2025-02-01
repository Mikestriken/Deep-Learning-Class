# %%
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

from API_Helpers import *

X_train = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz")[0x10:]
Y_train = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz")[0x10:]
Y_test = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]

plt.imshow(X_train.reshape(-1,28,28)[0]), Y_train[0]
plt.show()
# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
# %%
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

Y_dataset = torch.arange(start=0, end=1.125, step=0.125).unsqueeze(dim=1)
X_dataset = torch.arange(start=0, end=len(Y_dataset)).unsqueeze(dim=1)

Y_Train = Y_dataset[:int(0.8*len(Y_dataset))]
Y_Test = Y_dataset[int(0.8*len(Y_dataset)):]
X_Train = X_dataset[:int(0.8*len(X_dataset))]
X_Test = X_dataset[int(0.8*len(X_dataset)):]

def plotPredictions(predictions:torch.Tensor = None):
    plt.scatter(x=X_Train,y=Y_Train, c='b')
    plt.scatter(x=X_Test,y=Y_Test, c='g')
    
    if predictions != None:
        assert predictions.shape == X_Test.shape, f"predictions shape mismatch\npredictions: {predictions.shape}\nX_Test: {X_Test.shape}"
        plt.scatter(x=X_Test, y=predictions, c='r')
    plt.plot()
    plt.show()


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(1),requires_grad=True)
        
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return (self.weight*x + self.bias).relu()

torch.manual_seed("42")    
M1 = LinearModel()

M1.state_dict()

with torch.inference_mode():
    Y_pred = M1(X_Test)
    plotPredictions(Y_pred)

# %% 
Loss_Function:nn.MSELoss = nn.MSELoss()

Optimizer_Function:torch.optim.SGD = torch.optim.SGD(params=M1.parameters(),
                                                     lr=0.01)

epochs = 250

Y_pred = M1(X_Train)
print(torch.sum((Y_pred-Y_Train)**2).data.item())
print(f"before: {M1.state_dict()}")
MSELoss = []
for epoch in range(epochs):
    M1.train()
    Y_pred = M1(X_Train)
    loss:nn.MSELoss = Loss_Function(Y_pred, Y_Train)
    
    MSELoss += [torch.sum((Y_pred-Y_Train)**2).data.item()]
    print(torch.sum((Y_pred-Y_Train)**2).data.item())
    
    Optimizer_Function.zero_grad()
    
    loss.backward()
    
    Optimizer_Function.step()
    
    M1.eval()

MSELoss += [torch.sum((Y_pred-Y_Train)**2).data.item()]
print(torch.sum((Y_pred-Y_Train)**2).data.item())
print(f"after: {M1.state_dict()}")
    
with torch.inference_mode():
    Y_pred = M1(X_Test)
    plotPredictions(Y_pred)

plt.scatter(x=[i for i in range(len(MSELoss))], y=[element*100 for element in MSELoss])
plt.ylim(0, 2)
plt.plot()
plt.show()