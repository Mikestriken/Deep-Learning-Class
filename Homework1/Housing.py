# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('./Housing.csv')

le = LabelEncoder()

df['mainroad'] = le.fit_transform(df['mainroad'])
df['guestroom'] = le.fit_transform(df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])

useOnHotEncoding: bool = True
if useOnHotEncoding:
    df = pd.get_dummies(df, columns=['furnishingstatus'], dtype=int)
else:
    df = df.drop(['furnishingstatus'], axis=1)

# print(df.head())

Y_data = np.log1p(df.pop('price'))
scaler = StandardScaler()
X_data = scaler.fit_transform(df)

# print(pd.DataFrame(scaler.inverse_transform(X_data)))
# print(pd.DataFrame(np.expm1(Y_data)))

X_data = torch.tensor(X_data, dtype=torch.float32, device="cuda")
Y_data = torch.tensor(Y_data, dtype=torch.float32, device="cuda")

# print(X_data)
# print(Y_data)

class HouseDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

train_dataset = HouseDataset(X_train, Y_train)
test_dataset = HouseDataset(X_test, Y_test)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

def getAccuracy(labelSet: torch.tensor, predSet:torch.tensor) -> float:
    correct = torch.eq(labelSet, predSet).sum().item()
    accuracy = (correct/len(predSet)) * 100
    return accuracy

model = nn.Sequential(
    nn.Linear(in_features=len(X_train[0]), out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=1),
).to(torch.device("cuda"))
# %%
# ===============================================================================================================
#                                                   Load Model
# ===============================================================================================================
model = torch.load('Saved_Models/Housing.pt')  
# %% 
# ===============================================================================================================
#                                                    Training
# ===============================================================================================================
Loss_Function = nn.MSELoss()
Optimizer_Function = torch.optim.SGD(params=model.parameters(),
                                     lr=0.0005)

EPOCHS = 1000
epochIterator = 0

avgTrainBatchLossPerEpoch = []
avgTestBatchLossPerEpoch = []

while (epochIterator < EPOCHS or EPOCHS == -1):
    startTime = time.time()
    model.train()
    
    totalLossInEpoch = 0
    for X_train_batch, Y_train_batch in train_loader:
        Y_train_pred_logits:torch.Tensor = model(X_train_batch)
        trainBatchLoss = Loss_Function(Y_train_pred_logits, Y_train_batch)
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        totalLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        trainEpochAverageBatchLoss = totalLossInEpoch/len(train_loader)
        avgTrainBatchLossPerEpoch += [trainEpochAverageBatchLoss]
        
        totalLossInEpoch = 0
        for X_test_batch_imgs, Y_test_batch_labels in test_loader:
            Y_test_pred_imgs_logits:torch.Tensor = model(X_test_batch_imgs)
            testBatchLoss = Loss_Function(Y_train_pred_logits, Y_train_batch)
            
            totalLossInEpoch += testBatchLoss
            
        testEpochAverageBatchLoss = totalLossInEpoch/len(test_loader)
        avgTestBatchLossPerEpoch += [testEpochAverageBatchLoss]
        
    
        epochTime = time.time() - startTime
        estRemainingTime = (EPOCHS - epochIterator)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {trainEpochAverageBatchLoss:.5f}\t| test loss: {testEpochAverageBatchLoss:.5f}\t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        epochIterator += 1
# %%
# ===============================================================================================================
#                                                   Plot Loss
# ===============================================================================================================
with torch.inference_mode():
    avgTrainBatchLossPerEpoch = torch.tensor(avgTrainBatchLossPerEpoch).cpu()
    avgTestBatchLossPerEpoch = torch.tensor(avgTestBatchLossPerEpoch).cpu()
    
    # plt.scatter(x=[x for x in range(len(avgTrainBatchLossPerEpoch))], y=avgTrainBatchLossPerEpoch)
    # plt.scatter(x=[x for x in range(len(avgTestBatchLossPerEpoch))], y=avgTestBatchLossPerEpoch)
    plt.plot(avgTrainBatchLossPerEpoch, label="Training Loss", alpha=0.9, zorder=2)
    plt.plot(avgTestBatchLossPerEpoch, label="Test/Validation Loss", alpha=0.7, zorder=1)
    
    
    plt.plot(avgTrainBatchLossPerEpoch + 0.15, label="Training Loss + 0.15", c='b', linestyle='--', alpha=0.9, zorder=2)
    plt.plot(avgTrainBatchLossPerEpoch - 0.1, label="Training Loss - 0.10", c='b', linestyle='--', alpha=0.9, zorder=2)

    
    plt.title('Loss Per Epoch; Increased Complexity')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.ylim(0, 0.4)

    # Adjust layout and display the plot
    plt.tight_layout()  # Avoid overlap between subplots
    plt.plot()
    plt.show()

# %%
# ===============================================================================================================
#                                                   Save Model
# ===============================================================================================================
torch.save(model, 'Saved_Models/Housing_Complex.pt')

# %%
# ===============================================================================================================
#                                                 Visualize Data
# ===============================================================================================================
    
with torch.inference_mode():
    modelLogits = model(test_loader.dataset.features)
    modelOutputs = np.expm1(modelLogits.cpu())
    
    outputIndex = 0
    print(f"Output: {modelLogits[outputIndex]}")
    print(f"Actual Label: {Y_test[outputIndex]}")
    # print(f"Output Label Array:\n{torch.squeeze(modelOutputs)}")
    # print(f"Actual Label Array:\n{np.expm1(Y_test.cpu())}")
# %%
