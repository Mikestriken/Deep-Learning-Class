# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
from torchtnt.utils.data import CudaDataPrefetcher
from torchvision import datasets as imageDatasets, transforms as imageTransforms

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

assert torch.cuda.is_available(), "ERR: No GPU available"

DEVICE:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available(): torch.cuda.empty_cache()

charset_file_path:Path = Path(__file__).parent / "problem1CharSet.txt"
dataset:str = charset_file_path.read_text(encoding="utf-8")

chars = sorted(list(set(dataset)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)} 

# Preparing the dataset
SEQUENCE_LENGTH:int = 20
X = []
y = []
for i in range(len(dataset) - SEQUENCE_LENGTH):
    sequence = dataset[i:i + SEQUENCE_LENGTH]
    label = dataset[i + SEQUENCE_LENGTH]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

class CharDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Splitting the dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CharDataset(X_train, y_train)
test_dataset = CharDataset(X_test, y_test)

# Converting data to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.long)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_test = torch.tensor(X_test, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)

NUM_WORKERS:int = 0 #int(os.cpu_count() / 2)
NUM_BATCHES_TO_PREFETCH:int = 2

train_loader:data.DataLoader = data.DataLoader(train_dataset, batch_size=32,
                                            shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

test_loader:data.DataLoader = data.DataLoader(test_dataset, batch_size=32,
                                            shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

train_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=train_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)
test_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=test_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)

DROPOUT_PROB:float = 0.0

NUM_CHARS:int = len(chars)

class RNNNet(nn.Module):
    class lstmIndices(Enum):
        OUTPUT = 0
    
    def __init__(self, useEmbedding:bool = True):
        super().__init__()
        
        self.useEmbedding:bool = useEmbedding
        
        self.charEmbeddingLayer:nn.Embedding = nn.Embedding(num_embeddings=NUM_CHARS, embedding_dim=128)
        
        self.lstm:nn.LSTM = nn.LSTM(input_size=128, hidden_size=128,
                                    batch_first=True, dropout=DROPOUT_PROB)
        
        self.outputDenseLayer = nn.Sequential(
            
            nn.Linear(in_features=128, out_features=NUM_CHARS),
        )
    
    def forward(self, x):
        # x = self.charEmbeddingLayer(x)
        # x = self.lstm(x)[self.lstmIndices.OUTPUT][:, -1, :]
        # return self.outputDenseLayer(x)
        
        if self.useEmbedding:
            return self.outputDenseLayer(
                self.lstm(
                    self.charEmbeddingLayer(x)
                )[self.lstmIndices.OUTPUT.value][:, -1, :] # next predicted character output at end of sequence
            )
        else:
            return self.outputDenseLayer(
                self.lstm(
                    x
                )[self.lstmIndices.OUTPUT.value][:, -1, :] # next predicted character output at end of sequence
            )

# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))

# ========== Model Parameters ==========
model:RNNNet = RNNNet(useEmbedding=False).to(DEVICE) # calflops doesn't work with embedding layer

from calflops import calculate_flops
from torchvision import models
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(32, SEQUENCE_LENGTH, 128),
                                      output_as_string=True,
                                      output_precision=4)
print(f"Params: {params}\nFLOPS (Floating-Point Operations Per Second): {flops}\nMACS (Multiply–ACcumulate operation per Second): {macs}\n")



model:RNNNet = RNNNet(useEmbedding=True).to(DEVICE)
print(f"Total Num Params in loaded model: {sum(p.numel() for p in model.parameters())}")
# %%
# ================================================ Shape Testing ================================================
# X:torch.Tensor = torch.rand(size=(32, 32, 32, 3)).permute(0, 3, 1, 2)

# X:torch.Tensor = torch.tensor([[2, 1, 3, 4, 5, 2, 8, 1, 2, 0],
#                                [2, 1, 3, 4, 5, 2, 8, 1, 2, 0],
#                                [2, 1, 3, 4, 5, 2, 8, 1, 2, 0]])

# print(X.shape)
# print(model(X.to(DEVICE)))

# firstBatch = next(iter(train_loader))
# X, Y = firstBatch

# print(X.shape)
# print(X[0])

# print(model(X.to(DEVICE))) # CUDA error: device-side assert triggered

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
Optimizer_Function:torch.optim.Adam = torch.optim.Adam(params=model.parameters())#,
                                                    #  lr=0.15) #0.15

EPOCHS:int = 50
epochIterator:int = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

bestTestAccuracy:float = 0

while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + linearOffset(input=testEpochAccuracy, offset=3, target=99) or bestTestAccuracy < 50):
    startTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch, Y_train_batch in train_prefetcher: #==
        X_train_batch:torch.Tensor = X_train_batch.to(DEVICE, non_blocking=True)
        Y_train_batch:torch.Tensor = Y_train_batch.to(DEVICE, non_blocking=True)
        
        Y_train_pred_logits:torch.Tensor = model(X_train_batch)
        
        trainBatchLoss:float = Loss_Function(Y_train_pred_logits, Y_train_batch.type(torch.int64))
        
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
        for X_test_batch, Y_test_batch in test_prefetcher: #==
            X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
            Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
        
            Y_test_pred_logits:torch.Tensor = model(X_test_batch)
            Y_test_pred:torch.Tensor = Y_test_pred_logits.softmax(dim=1).argmax(dim=1)
        
            testBatchLoss:float = Loss_Function(Y_test_pred_logits, Y_test_batch.type(torch.int64))
    
            numCorrectInEpoch += torch.eq(Y_test_pred, Y_test_batch).sum().item()
            
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