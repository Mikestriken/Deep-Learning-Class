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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.arange(0, 3).to(device)
# %%
