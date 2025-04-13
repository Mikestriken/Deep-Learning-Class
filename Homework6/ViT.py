# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtnt.utils.data import CudaDataPrefetcher
from torchvision import datasets as imageDatasets, transforms as imageTransforms
from torchprofile import profile_macs

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
import matplotlib.pyplot as plt

import os
from pathlib import Path
import time
import signal
from enum import Enum
import json
from itertools import takewhile

assert torch.cuda.is_available(), "ERR: No GPU available"

DEVICE:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available(): torch.cuda.empty_cache()

data_transform = imageTransforms.Compose([
    imageTransforms.TrivialAugmentWide(),
    imageTransforms.ToTensor(),
    imageTransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

NUM_WORKERS:int = 0 #int(os.cpu_count() / 2)
NUM_BATCHES_TO_PREFETCH:int = 2

train_loader:data.DataLoader = data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

test_loader:data.DataLoader = data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

train_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=train_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)
test_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=test_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)
# Patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x:torch.Tensor):
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(start_dim=2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropoutP=0.1):
        super().__init__()
        self.layer_norm1:nn.LayerNorm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attention:nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropoutP, batch_first=True)
        self.layer_norm2:nn.LayerNorm = nn.LayerNorm(normalized_shape=embed_dim)
        
        self.mlp:nn.Sequential = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(dropoutP),
            nn.Linear(in_features=mlp_dim, out_features=embed_dim),
            nn.Dropout(dropoutP)
        )
        
    def forward(self, x):
        x2 = self.layer_norm1(x)
        attention_output, _ = self.attention(x2, x2, x2)
        
        x = x + attention_output
        
        x2 = self.layer_norm2(x)
        mlp_output = self.mlp(x2)
        
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        NUM_LAYERS:int = 4
        NUM_HEADS:int = 4
        EMBEDDING_SIZE:int = 2048
        # HIDDEN_SIZE:int = 2048
        DROPOUT_PROB:float = 0.1
        
        IMAGE_SIZE:int = 32
        PATCH_SIZE:int = 4
        NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
        
        self.patch_embed:PatchEmbedding = PatchEmbedding(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBEDDING_SIZE)
        self.CLS_TOKEN:nn.Parameter = nn.Parameter(torch.zeros(1, 1, EMBEDDING_SIZE))
        self.pos_embed:nn.Parameter = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, EMBEDDING_SIZE))
        
        self.dropout:nn.Dropout = nn.Dropout(DROPOUT_PROB)
        
        self.transformer:nn.ModuleList = nn.ModuleList(
            [TransformerEncoder(embed_dim=EMBEDDING_SIZE, num_heads=NUM_HEADS, mlp_dim=4*EMBEDDING_SIZE, dropoutP=DROPOUT_PROB) for _ in range(NUM_LAYERS)]
        )
        
        self.layer_norm:nn.LayerNorm = nn.LayerNorm(normalized_shape=EMBEDDING_SIZE)
        self.classifier:nn.Linear = nn.Linear(in_features=EMBEDDING_SIZE, out_features=100)

    def forward(self, X):
        BATCH_SIZE = X.shape[0]
        
        X = self.patch_embed(X) # (B, 3, H', W') → (B, num_patches, embed_dim)
        
        cls_tokens = self.CLS_TOKEN.expand(BATCH_SIZE, -1, -1) # Copy reference to CLS_TOKEN. (1, 1, CLS_TOKEN) → (BATCH_SIZE, 1, CLS_TOKEN)
        X = torch.cat((cls_tokens, X), dim=1) # Prepend cls_tokens to num_patches in each batch
        X = X + self.pos_embed
        X = self.dropout(X)
        
        # X => (B, num_patches + <CLS>, embed_dim)
        for transformer in self.transformer:
            X = transformer(X)
            
        X = self.layer_norm(X)
        cls_token_final = X[:, 0] # For all batches, get CLS_TOKEN
        X = self.classifier(cls_token_final)
        return X

# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))

# ========== Model Parameters ==========
model:VisionTransformer = VisionTransformer().to(DEVICE)
total_params = sum([p.numel() for p in model.parameters()])
print(f"Total Num Params in loaded model: {total_params:,}")

# Calculate MACs (Multiply-Accumulate Operations)
# Create sample inputs for profiling
firstBatch = next(iter(train_loader))
sample_X, sample_Y = firstBatch
sample_X, sample_Y = sample_X.to(DEVICE), sample_Y.to(DEVICE)

# Profile the model
macs = profile_macs(model, (sample_X, ))
print(f"Computational complexity: {macs:,} MACs")
print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")
# %%
# ================================================ Shape Testing ================================================
firstBatch = next(iter(test_loader))
X, Y = firstBatch
X, Y = X.to(DEVICE), Y.to(DEVICE)

print(f"X: {X.shape}")
logits = model(X.to(DEVICE))
print(f"Logits: {logits.shape}")
print(f"Pred: {logits.argmax(dim=1).shape}")
print(f"Expected: {Y.shape}")

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

def linearOffset(input, offset, target):
    # max() ensures offset is always positive or 0
    # min() returns the smaller offset between target - input and default offset
    return max(0, min(offset, target - input))


Loss_Function:nn.CrossEntropyLoss = nn.CrossEntropyLoss()
# Optimizer_Function:torch.optim.Adam = torch.optim.Adam(params=model.parameters())
# Optimizer_Function:torch.optim.SGD = torch.optim.SGD(params=model.parameters(),
#                                                      lr=0.0001)
Optimizer_Function:torch.optim.Adam = torch.optim.Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-5
)

EPOCHS:int = 50
epochIterator:int = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

bestTestAccuracy:float = 0

MINIMUM_TEST_ACCURACY:int = 0
SAVE_CHECKPOINTS:bool = False

trainStartTime:float = time.time()
while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + linearOffset(input=testEpochAccuracy, offset=3, target=99) or bestTestAccuracy < MINIMUM_TEST_ACCURACY):
    epochStartTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch, Y_train_batch in train_prefetcher:
        X_train_batch:torch.Tensor = X_train_batch.to(DEVICE, non_blocking=True)
        Y_train_batch:torch.Tensor = Y_train_batch.to(DEVICE, non_blocking=True)
        
        Y_train_pred_logits:torch.Tensor = model(X_train_batch)
        
        trainBatchLoss = Loss_Function(Y_train_pred_logits, Y_train_batch.type(torch.int64))
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        numCorrectInEpoch += torch.eq(Y_train_pred_logits.argmax(dim=1), Y_train_batch).sum().item()
        totalTrainLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        trainEpochAverageBatchLoss:float = totalTrainLossInEpoch/len(train_loader)
        avgTrainBatchLossPerEpoch += [trainEpochAverageBatchLoss]
        
        trainEpochAccuracy:float = numCorrectInEpoch/len(train_loader.dataset) * 100 # accuracy is calculated per item in a batch instead of per batch
        trainAccuracyPerEpoch += [trainEpochAccuracy]
        
        numCorrectInEpoch:int = 0
        totalTestLossInEpoch:float = 0
        for X_test_batch, Y_test_batch in test_prefetcher:
            X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
            Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
        
            Y_test_pred_logits:torch.Tensor = model(X_test_batch)
        
            testBatchLoss = Loss_Function(Y_test_pred_logits, Y_test_batch.type(torch.int64))
    
            numCorrectInEpoch += torch.eq(Y_test_pred_logits.argmax(dim=1), Y_test_batch).sum().item()
            
            totalTestLossInEpoch += testBatchLoss
        
        testEpochAverageBatchLoss:float = totalTestLossInEpoch/len(test_loader)
        avgTestBatchLossPerEpoch += [testEpochAverageBatchLoss]
        
        testEpochAccuracy:float = numCorrectInEpoch/len(test_loader.dataset) * 100
        testAccuracyPerEpoch += [testEpochAccuracy]
    
        epochTime:float = time.time() - epochStartTime
        estRemainingTime:float = (EPOCHS - epochIterator - 1)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {trainEpochAverageBatchLoss:.5f}, train accuracy: {trainEpochAccuracy:.2f}% \t| test loss: {testEpochAverageBatchLoss:.5f}, test accuracy: {testEpochAccuracy:.2f}% \t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        newBestModel:bool = testEpochAccuracy > MINIMUM_TEST_ACCURACY and testEpochAccuracy > bestTestAccuracy
        if newBestModel: 
            bestTestAccuracy:float = testEpochAccuracy
            print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑ NEW BEST MODEL ↑↑↑↑↑↑↑↑↑↑↑↑↑")
            
        if SAVE_CHECKPOINTS and newBestModel: 
            torch.save(model.state_dict(), 'Saved_Models/best_model.pth')
            print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ SAVED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
        
        epochIterator += 1
        
totalTrainTime:float = (time.time() - trainStartTime)/60
averageEpochTime:float = totalTrainTime / epochIterator

print(f"Total Training Time: {int(totalTrainTime):02}:{int((totalTrainTime - int(totalTrainTime))*60):02}")
print(f"Average Epoch Time: {int(averageEpochTime):02}:{int((averageEpochTime - int(averageEpochTime))*60):02}")
        
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
# torch.save(model, 'Saved_Models/ResNet11_Baseline_CIFAR10.pt')

# %%
# ===============================================================================================================
#                                                 Visualize Data
# ===============================================================================================================

# Accuracy = #Correct / #Predictions
def getAccuracy(labelSet: torch.tensor, predSet:torch.tensor) -> float:
    correct = torch.eq(labelSet, predSet).sum().item()
    accuracy = (correct/len(predSet)) * 100
    return accuracy

with torch.inference_mode():
    torch.cuda.empty_cache()
    for X_test_batch, Y_test_batch in test_loader:
        X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
        Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
            
        modelLogits = model(X_test_batch)
        modelOutputs = modelLogits.softmax(dim=1).argmax(dim=1)
        
        outputIndex = 0
        print(f"Output Logit: {modelLogits[outputIndex]}")
        print(f"Expected Label: {torch.argmax(modelLogits[outputIndex])}")
        print(f"Actual Label: {Y_test_batch[outputIndex]}")
        print(f"Output Label Array: {modelOutputs}")
        print(f"Actual Label Array: {Y_test_batch.type(torch.int64)}")
        
        print(f"Accuracy: {getAccuracy(labelSet=Y_test_batch.type(torch.int64), predSet=modelOutputs):.2f}%")
        
        precision = precision_score(Y_test_batch.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        recall = recall_score(Y_test_batch.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        f1 = f1_score(Y_test_batch.type(torch.int64).cpu(), modelOutputs.cpu(), average='macro')
        confusion = confusion_matrix(Y_test_batch.type(torch.int64).cpu(), modelOutputs.cpu())
        
        print(f'Precision: {(precision * 100):.4f}%')
        print(f'Recall: {(recall * 100):.4f}%')
        print(f'F1 Score: {(f1 * 100):.4f}%')
        print(f'Confusion Matrix:\n{confusion}')
        
        confusionPlot = ConfusionMatrixDisplay(confusion_matrix=confusion)
        confusionPlot.plot()
# %%
# %%