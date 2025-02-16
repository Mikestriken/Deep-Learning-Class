# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets as imageDatasets, transforms as imageTransforms

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
import matplotlib.pyplot as plt

import os
import time
import signal

# from API_Helpers import *

assert torch.cuda.is_available(), "ERR: No GPU available"

DEVICE:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS:int = 0 #int(os.cpu_count() / 2)

if torch.cuda.is_available(): torch.cuda.empty_cache()

data_transform = imageTransforms.Compose([
    imageTransforms.TrivialAugmentWide(),
    imageTransforms.ToTensor()
])

USE_CIFAR100:bool = True

if USE_CIFAR100:
    train_dataset:imageDatasets.CIFAR100 = imageDatasets.CIFAR100(root='./datasets', train=True,
                                                                  download=True, transform=data_transform)
    test_dataset:imageDatasets.CIFAR100 = imageDatasets.CIFAR100(root='./datasets', train=False,
                                                                 download=True, transform=data_transform)
    
else:
    train_dataset:imageDatasets.CIFAR10 = imageDatasets.CIFAR10(root='./datasets', train=True,
                                                                download=True, transform=data_transform)
    test_dataset:imageDatasets.CIFAR10 = imageDatasets.CIFAR10(root='./datasets', train=False,
                                                               download=True, transform=data_transform)

train_loader:data.DataLoader = data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

test_loader:data.DataLoader = data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ===================== Preview Images =====================
# IMAGE_INDEX = 0
# labels = train_dataset.classes
# label = labels[train_dataset[IMAGE_INDEX][1]]
# img = (train_dataset[IMAGE_INDEX][0].permute(1,2,0).numpy() * 255).astype(np.uint8)

# print(f"List of Classes:\n{labels}")
# print(f"Number of Classes: {len(labels)}")
# print(f"Image Class: {label}")
# plt.imshow(img)
# plt.show()
# ===========================================================

# Accuracy = #Correct / #Predictions
def getAccuracy(labelSet: torch.tensor, predSet:torch.tensor) -> float:
    correct = torch.eq(labelSet, predSet).sum().item()
    accuracy = (correct/len(predSet)) * 100
    return accuracy

DROPOUT_PROB:float = 0.2

def getConvBlock(in_channels:int, 
                 out_channels:int, 
                 convKernel:int=3, 
                 convStride:int=1, 
                 convPadding:int=1, 
                 poolKernel:int=2, 
                 poolPadding:int=0, 
                 poolStride:int=2, 
                 disablePool:bool = False,
                 disableBatchNorm:bool = False,
                 disableDropout:bool = False,
                 
                 ) -> nn.Sequential:
    
    convBlock:list = [
        nn.Conv2d(in_channels=in_channels,
                    kernel_size=convKernel,             # Default kernel_size=3
                    stride=convStride,                  # Default stride=1
                    padding=convPadding,                # Default padding=1
                    out_channels=out_channels,
                    bias=disableBatchNorm),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ]
    
    if not disablePool: convBlock.append(nn.MaxPool2d(kernel_size=poolKernel, padding=poolPadding, stride=poolStride))
    if not disableDropout: convBlock.append(nn.Dropout2d(DROPOUT_PROB))
    
    return nn.Sequential(*convBlock)

class ResNetBlock(nn.Module):
    def __init__(self, channels:int, 
                       convKernel:int=3, 
                       convStride:int=1, 
                       convPadding:int=1, 
                       disableBatchNorm:bool = False,
                       disableDropout:bool = False,
                       DROPOUT_PROB:float = 0.2):
        
        super().__init__()
        
        self.mainLayers:nn.Sequential = nn.Sequential(
            getConvBlock(in_channels=channels,   out_channels=channels,
                         disablePool=True, 
                         disableDropout=disableDropout), # Out: 32 x 32
            
            getConvBlock(in_channels=channels,   out_channels=channels,
                         disablePool=True, 
                         disableDropout=disableDropout) # Out: 16 x 16 (Pooling => 1/2)
        )
        
        # Note: In practice the 1x1 conv block had a massively negative impact on training
        # self.residualLayers:nn.Sequential = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels,
        #                 kernel_size=1,
        #                 stride=convStride,
        #                 padding=0,
        #                 out_channels=out_channels),
        # )
        
    def forward(self, x):
        return self.mainLayers(x) + x

class ResNet(nn.Module):
    def __init__(self, numConvBlocks:int):
        super().__init__()
        
        assert numConvBlocks == 11 or numConvBlocks == 18, "Invalid number of conv blocks!"
        
        DISABLE_DROPOUT:bool = True
        
        if numConvBlocks == 11:
            resNetBlocks:nn.Sequential = nn.Sequential(
                ResNetBlock(channels=128,   # 64 → 64
                            disableDropout=DISABLE_DROPOUT), # Out: 16 x 16
                
                getConvBlock(in_channels=128,   out_channels=256,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 8 x 8 (Pooling => 1/2)
                
                getConvBlock(in_channels=256,   out_channels=512,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 4 x 4 (Pooling => 1/2)
                
                ResNetBlock(channels=512,  # 64 → 128
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=512,   out_channels=1028,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 2 x 2 (Pooling => 1/2)
                
                ResNetBlock(channels=1028,  # 128 → 256
                            disableDropout=DISABLE_DROPOUT),
                
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # Out: 1 x 1 (Pooling => 1/2)
            )
        elif numConvBlocks == 18:
            resNetBlocks:nn.Sequential = nn.Sequential(
                ResNetBlock(channels=128,   # 64 → 64
                            disableDropout=DISABLE_DROPOUT), # Out: 16 x 16
                
                getConvBlock(in_channels=128,   out_channels=256,
                            disablePool=True, 
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=256,   out_channels=512,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 8 x 8 (Pooling => 1/2)
                
                ResNetBlock(channels=512,  # 64 → 128
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=512,   out_channels=512,
                            disablePool=True, 
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=512,   out_channels=1028,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 4 x 4 (Pooling => 1/2)
                
                ResNetBlock(channels=1028,  # 64 → 128
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=1028,   out_channels=2056,
                            disablePool=True, 
                            disableDropout=DISABLE_DROPOUT),
                
                ResNetBlock(channels=2056,  # 64 → 128
                            disableDropout=DISABLE_DROPOUT),
                
                getConvBlock(in_channels=2056,   out_channels=1028,
                            disablePool=False, 
                            disableDropout=DISABLE_DROPOUT), # Out: 2 x 2 (Pooling => 1/2)
                
                ResNetBlock(channels=1028,  # 128 → 256
                            disableDropout=DISABLE_DROPOUT),
                
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # Out: 1 x 1 (Pooling => 1/2)
            )
        
        self.model = nn.Sequential(
            getConvBlock(in_channels=3,   out_channels=64,
                         disablePool=True, 
                         disableDropout=DISABLE_DROPOUT), # Out: 32 x 32
            
            getConvBlock(in_channels=64,   out_channels=128,
                         disablePool=False, 
                         disableDropout=DISABLE_DROPOUT), # Out: 16 x 16 (Pooling => 1/2)
            
            resNetBlocks, # Out: 1 x 1
            
            nn.Flatten(),
            
            nn.Linear(in_features=1028, out_features=(10 if not USE_CIFAR100 else 100)),
        )
    
    def forward(self, x):
        return self.model(x)

model:ResNet = ResNet(18).to(DEVICE)
# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))
print(f"Total Num Params in loaded model: {sum(p.numel() for p in model.parameters())}")
# %%
# ================================================ Shape Testing ================================================

input:torch.Tensor = torch.rand(size=(32, 32, 32, 3)).permute(0, 3, 1, 2)
firstBatch = next(iter(test_loader))

X, Y = firstBatch
print(X.shape)
print(model(X.to(DEVICE)))

# %%
# ===============================================================================================================
#                                                   Load Model
# ===============================================================================================================
# model = torch.load('Saved_Models/20_Epoch_CIFAR.pt')
# %% 
# ===============================================================================================================
#                                                    Training
# ===============================================================================================================
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("Interrupt received. Flag set...")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

Loss_Function:nn.CrossEntropyLoss = nn.CrossEntropyLoss()
Optimizer_Function:torch.optim.SGD = torch.optim.SGD(params=model.parameters(),
                                                     lr=0.15) #0.15

EPOCHS:int = 50
epochIterator:int = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

bestTestAccuracy:float = 0

while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + max(0, min(0, 99 - testEpochAccuracy)) or bestTestAccuracy < 55):
    startTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch_imgs, Y_train_batch_labels in train_loader:
        X_train_batch_imgs:torch.Tensor = X_train_batch_imgs.to(DEVICE, non_blocking=True)
        Y_train_batch_labels:torch.Tensor = Y_train_batch_labels.to(DEVICE, non_blocking=True)
        
        Y_train_pred_imgs_logits:torch.Tensor = model(X_train_batch_imgs)
        
        trainBatchLoss:float = Loss_Function(Y_train_pred_imgs_logits, Y_train_batch_labels.type(torch.int64))
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        numCorrectInEpoch += torch.eq(Y_train_pred_imgs_logits.argmax(dim=1), Y_train_batch_labels).sum().item()
        totalTrainLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        trainEpochAverageBatchLoss:float = totalTrainLossInEpoch/len(train_loader)
        avgTrainBatchLossPerEpoch += [trainEpochAverageBatchLoss]
        
        trainEpochAccuracy:float = numCorrectInEpoch/len(train_loader.dataset) * 100 # accuracy is calculated per item in a batch instead of per batch
        trainAccuracyPerEpoch += [trainEpochAccuracy]
        
        numCorrectInEpoch:int = 0
        totalTestLossInEpoch:float = 0
        for X_test_batch_imgs, Y_test_batch_labels in test_loader:
            X_test_batch_imgs:torch.Tensor = X_test_batch_imgs.to(DEVICE, non_blocking=True)
            Y_test_batch_labels:torch.Tensor = Y_test_batch_labels.to(DEVICE, non_blocking=True)
        
            Y_test_pred_imgs_logits:torch.Tensor = model(X_test_batch_imgs)
            Y_test_pred_imgs_labels:torch.Tensor = Y_test_pred_imgs_logits.softmax(dim=1).argmax(dim=1)
        
            testBatchLoss:float = Loss_Function(Y_test_pred_imgs_logits, Y_test_batch_labels.type(torch.int64))
    
            numCorrectInEpoch += torch.eq(Y_test_pred_imgs_labels, Y_test_batch_labels).sum().item()
            
            totalTestLossInEpoch += testBatchLoss
        
        testEpochAverageBatchLoss:float = totalTestLossInEpoch/len(test_loader)
        avgTestBatchLossPerEpoch += [testEpochAverageBatchLoss]
        
        testEpochAccuracy:float = numCorrectInEpoch/len(test_loader.dataset) * 100
        testAccuracyPerEpoch += [testEpochAccuracy]
    
        epochTime:float = time.time() - startTime
        estRemainingTime:float = (EPOCHS - epochIterator - 1)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {trainEpochAverageBatchLoss:.5f}, train accuracy: {trainEpochAccuracy:.2f}% \t| test loss: {testEpochAverageBatchLoss:.5f}, test accuracy: {testEpochAccuracy:.2f}% \t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        if testEpochAccuracy > 50 and testEpochAccuracy > bestTestAccuracy: 
            bestTestAccuracy:float = testEpochAccuracy
            torch.save(model.state_dict(), 'Saved_Models/best_model.pth')
            print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ SAVED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
        
        epochIterator += 1
        
# %%
# ===============================================================================================================
#                                                   Plot Loss
# ===============================================================================================================
with torch.inference_mode():
    avgTrainBatchLossPerEpoch1:list = torch.tensor(avgTrainBatchLossPerEpoch).cpu()
    avgTestBatchLossPerEpoch1:list = torch.tensor(avgTestBatchLossPerEpoch).cpu()
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # First subplot
    axs[0].scatter(x=[x for x in range(len(avgTrainBatchLossPerEpoch1))], y=avgTrainBatchLossPerEpoch1, label="Training Loss")
    axs[0].scatter(x=[x for x in range(len(avgTestBatchLossPerEpoch1))], y=avgTestBatchLossPerEpoch1, label="Test / Validation Loss")
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
torch.save(model, 'Saved_Models/ResNet11_Baseline_CIFAR10.pt')

# %%
# ===============================================================================================================
#                                                 Visualize Data
# ===============================================================================================================
    
with torch.inference_mode():
    torch.cuda.empty_cache()
    for X_test_batch_imgs, Y_test_batch_labels in test_loader:
        X_test_batch_imgs:torch.Tensor = X_test_batch_imgs.to(DEVICE, non_blocking=True)
        Y_test_batch_labels:torch.Tensor = Y_test_batch_labels.to(DEVICE, non_blocking=True)
            
        modelLogits = model(X_test_batch_imgs)
        modelOutputs = modelLogits.softmax(dim=1).argmax(dim=1)
        
        outputIndex = 0
        print(f"Output Logit: {modelLogits[outputIndex]}")
        print(f"Expected Label: {torch.argmax(modelLogits[outputIndex])}")
        print(f"Actual Label: {Y_test_batch_labels[outputIndex]}")
        print(f"Output Label Array: {modelOutputs}")
        print(f"Actual Label Array: {Y_test_batch_labels.type(torch.int64)}")
        
        print(f"Accuracy: {getAccuracy(labelSet=Y_test_batch_labels.type(torch.int64), predSet=modelOutputs):.2f}%")
        
        precision = precision_score(Y_test_batch_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        recall = recall_score(Y_test_batch_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        f1 = f1_score(Y_test_batch_labels.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        confusion = confusion_matrix(Y_test_batch_labels.type(torch.int64).cpu(), modelOutputs.cpu())
        
        print(f'Precision: {(precision * 100):.4f}%')
        print(f'Recall: {(recall * 100):.4f}%')
        print(f'F1 Score: {(f1 * 100):.4f}%')
        print(f'Confusion Matrix:\n{confusion}')
        
        confusionPlot = ConfusionMatrixDisplay(confusion_matrix=confusion)
        confusionPlot.plot()
# %%