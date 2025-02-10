# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from API_Helpers import *

script_dir = os.path.dirname(os.path.abspath(__file__))

torchvision.datasets.CIFAR10(root='./datasets', train=True,
                             download=True, transform=None)
torchvision.datasets.CIFAR10(root='./datasets', train=False,
                             download=True, transform=None)

batchMeta = os.path.join(script_dir, 'datasets/cifar-10-batches-py/batches.meta')
batch1 = os.path.join(script_dir, 'datasets/cifar-10-batches-py/data_batch_1')
batch2 = os.path.join(script_dir, 'datasets/cifar-10-batches-py/data_batch_2')
batch3 = os.path.join(script_dir, 'datasets/cifar-10-batches-py/data_batch_3')
batch4 = os.path.join(script_dir, 'datasets/cifar-10-batches-py/data_batch_4')
batch5 = os.path.join(script_dir, 'datasets/cifar-10-batches-py/data_batch_5')
batchTest = os.path.join(script_dir, 'datasets/cifar-10-batches-py/test_batch')

batch1_imgs = unpickle(batch1)["data"]
batch2_imgs = unpickle(batch2)["data"]
batch3_imgs = unpickle(batch3)["data"]
batch4_imgs = unpickle(batch4)["data"]
batch5_imgs = unpickle(batch5)["data"]
batchTest_imgs = unpickle(batchTest)["data"]

X_imgs = np.append(np.append(np.append(np.append(np.append(batch1_imgs, batch2_imgs), batch3_imgs), batch4_imgs), batch5_imgs), batchTest_imgs)

batchMeta_classification = unpickle(batchMeta)["label_names"]
batch1_labels = unpickle(batch1)["labels"]
batch2_labels = unpickle(batch2)["labels"]
batch3_labels = unpickle(batch3)["labels"]
batch4_labels = unpickle(batch4)["labels"]
batch5_labels = unpickle(batch5)["labels"]
batchTest_labels = unpickle(batchTest)["labels"]

Y_labels = np.append(np.append(np.append(np.append(np.append(batch1_labels, batch2_labels), batch3_labels), batch4_labels), batch5_labels), batchTest_labels)

X_imgs = torch.from_numpy(X_imgs).reshape(-1,3,32,32).permute(0,2,3,1).type(torch.float32)/255
Y_labels = torch.tensor(Y_labels, dtype=torch.float32)

# ========================== Train ==========================
X_imgs = X_imgs.reshape(-1,32*32*3)

Y_labels = Y_labels.to(torch.device("cuda"))
X_imgs = X_imgs.to(torch.device("cuda"))
# --------------------------- or ---------------------------
# IMAGE_INDEX = 1000
# print(batchMeta_classification)
# print(batchMeta_classification[Y_labels[IMAGE_INDEX].type(torch.int64)])
# plt.imshow(X_imgs[IMAGE_INDEX].cpu())
# plt.show()
# ===================== Preview Images =====================

class ImageDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X_train_imgs, X_test_imgs, Y_train_labels, Y_test_labels = train_test_split(X_imgs, Y_labels, test_size=0.2, random_state=42)

train_dataset = ImageDataset(X_train_imgs, Y_train_labels)
test_dataset = ImageDataset(X_test_imgs, Y_test_labels)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Accuracy = #Correct / #Predictions
def getAccuracy(labelSet: torch.tensor, predSet:torch.tensor) -> float:
    correct = torch.eq(labelSet, predSet).sum().item()
    accuracy = (correct/len(predSet)) * 100
    return accuracy

model = nn.Sequential(
    nn.Linear(in_features=32*32*3, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
    # nn.Softmax(dim=0)
).to(torch.device("cuda"))
# %%
# ===============================================================================================================
#                                                   Load Model
# ===============================================================================================================
model = torch.load('Saved_Models/20_Epoch_CIFAR.pt')
# %% 
# ===============================================================================================================
#                                                    Training
# ===============================================================================================================
Loss_Function = nn.CrossEntropyLoss()
Optimizer_Function = torch.optim.SGD(params=model.parameters(),
                                     lr=0.15)

EPOCHS = 20
epochIterator = 0

avgBatchLossPerEpoch = []
trainAccuracyPerEpoch = []
testAccuracyPerEpoch = []

while (epochIterator < EPOCHS or EPOCHS == -1):
    startTime = time.time()
    model.train()
    
    numCorrectInEpoch = 0
    totalLossInEpoch = 0
    for X_train_batch_imgs, Y_train_batch_labels in train_loader:
        Y_train_pred_imgs_logits:torch.Tensor = model(X_train_batch_imgs)
        
        trainBatchLoss = Loss_Function(Y_train_pred_imgs_logits, Y_train_batch_labels.type(torch.int64))
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        numCorrectInEpoch += torch.eq(Y_train_pred_imgs_logits.argmax(dim=1), Y_train_batch_labels).sum().item()
        totalLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        epochAverageBatchLoss = totalLossInEpoch/len(train_loader)
        avgBatchLossPerEpoch += [epochAverageBatchLoss]
        
        trainEpochAccuracy = numCorrectInEpoch/len(train_loader.dataset) * 100 # accuracy is calculated per item in a batch instead of per batch
        trainAccuracyPerEpoch += [trainEpochAccuracy]
        
        numCorrectInEpoch = 0
        for X_test_batch_imgs, Y_test_batch_labels in test_loader:
            Y_test_pred_imgs_logits:torch.Tensor = model(X_test_batch_imgs)
            Y_test_pred_imgs_labels:torch.Tensor = Y_test_pred_imgs_logits.softmax(dim=1).argmax(dim=1)
    
            numCorrectInEpoch += torch.eq(Y_test_pred_imgs_labels, Y_test_batch_labels).sum().item()
        
        testEpochAccuracy = numCorrectInEpoch/len(test_loader.dataset) * 100
        testAccuracyPerEpoch += [testEpochAccuracy]
    
        epochTime = time.time() - startTime
        estRemainingTime = (EPOCHS - epochIterator)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {epochAverageBatchLoss:.5f}, train accuracy: {trainEpochAccuracy:.2f}% \t| test accuracy: {testEpochAccuracy:.2f}% \t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        epochIterator += 1
        
# %%
# ===============================================================================================================
#                                                   Plot Loss
# ===============================================================================================================
with torch.inference_mode():
    avgBatchLossPerEpoch = torch.tensor(avgBatchLossPerEpoch).cpu()
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # First subplot
    axs[0].scatter(x=[x for x in range(len(avgBatchLossPerEpoch))], y=avgBatchLossPerEpoch, label="Training Loss")
    axs[0].set_title('Loss Per Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Second subplot
    axs[1].scatter(x=[x for x in range(len(trainAccuracyPerEpoch))], y=trainAccuracyPerEpoch, label="Training Accuracy")
    axs[1].scatter(x=[x for x in range(len(testAccuracyPerEpoch))], y=testAccuracyPerEpoch, label="Test / Validation Accuracy")
    axs[1].set_title('Accuracy Per Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy %')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adjust layout and display the plot
    plt.tight_layout()  # Avoid overlap between subplots
    plt.plot()
    plt.show()

# %%
# ===============================================================================================================
#                                                   Save Model
# ===============================================================================================================
# torch.save(model, 'Saved_Models/100_Epoch_CIFAR.pt')

# %%
# ===============================================================================================================
#                                                 Visualize Data
# ===============================================================================================================
    
with torch.inference_mode():
    modelLogits = model(test_loader.dataset.features)
    modelOutputs = modelLogits.softmax(dim=1).argmax(dim=1)
    
    outputIndex = 0
    print(f"Output Logit: {modelLogits[outputIndex]}")
    print(f"Expected Label: {torch.argmax(modelLogits[outputIndex])}")
    print(f"Actual Label: {Y_test_labels[outputIndex]}")
    print(f"Output Label Array: {modelOutputs}")
    print(f"Actual Label Array: {Y_test_labels.type(torch.int64)}")
    
    print(f"Accuracy: {getAccuracy(labelSet=Y_test_labels.type(torch.int64), predSet=modelOutputs):.2f}%")
    
    precision = precision_score(Y_test_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
    recall = recall_score(Y_test_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
    f1 = f1_score(Y_test_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
    confusion = confusion_matrix(Y_test_labels.type(torch.int64).cpu(), modelOutputs.cpu())
    
    print(f'Precision: {(precision * 100):.4f}%')
    print(f'Recall: {(recall * 100):.4f}%')
    print(f'F1 Score: {(f1 * 100):.4f}%')
    print(f'Confusion Matrix:\n{confusion}')
    
    confusionPlot = ConfusionMatrixDisplay(confusion_matrix=confusion)
    confusionPlot.plot()
# %%
