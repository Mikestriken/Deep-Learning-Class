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

jsonDatasets = None
trainSet_file_path:Path = Path(__file__).parent / "dataset.json"
with open(trainSet_file_path, 'r') as file:
    jsonDatasets = json.load(file)

# ========== Unique Words ==========
Train_dataset:np.array = np.array(jsonDatasets.get("english_to_french_train"))
Test_dataset:np.array = np.array(jsonDatasets.get("english_to_french_qualitative"))

english_words:set = set(word for sentence in Train_dataset[:,0] for word in sentence.split())
french_words:set = set(word for sentence in Train_dataset[:,1] for word in sentence.split())

english_words.update([word for sentence in Test_dataset[:,0] for word in sentence.split()])
french_words.update([word for sentence in Test_dataset[:,1] for word in sentence.split()])

english_words = sorted(list(english_words))
french_words = sorted(list(french_words))

PADDING_TOKEN = 0
EOS_TOKEN = 1
NUM_ADDED_TOKENS = 2

NUM_ENGLISH_WORDS:int = len(english_words) + NUM_ADDED_TOKENS
NUM_FRENCH_WORDS:int = len(french_words) + NUM_ADDED_TOKENS

english_word_to_ix = {"PAD": PADDING_TOKEN, "EOS": EOS_TOKEN, **{ch: i + NUM_ADDED_TOKENS for i, ch in enumerate(english_words)}} 
ix_to_english_word = {i: ch for ch, i in english_word_to_ix.items()}

french_word_to_ix = {"PAD": PADDING_TOKEN, "EOS": EOS_TOKEN, **{ch: i + NUM_ADDED_TOKENS for i, ch in enumerate(french_words)}} 
ix_to_french_word = {i: ch for ch, i in french_word_to_ix.items()}

# ========== Train Set ==========
X_Train = [[english_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Train_dataset[:,0]]
Y_Train = [[french_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Train_dataset[:,1]]

X_Train = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in X_Train], batch_first=True, padding_value=PADDING_TOKEN)
Y_Train = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in Y_Train], batch_first=True, padding_value=PADDING_TOKEN)

# ========== Test Set ==========
X_Test = [[english_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Test_dataset[:,0]]
Y_Test = [[french_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Test_dataset[:,1]]

X_Test = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in X_Test], batch_first=True, padding_value=PADDING_TOKEN)
Y_Test = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in Y_Test], batch_first=True, padding_value=PADDING_TOKEN)

class FrenchTranslationDataset(data.Dataset):
    def __init__(self, features, labels, padding_token = PADDING_TOKEN):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Splitting the dataset into training and validation sets
# X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Train_dataset:FrenchTranslationDataset = FrenchTranslationDataset(X_Train, Y_Train)
test_dataset:FrenchTranslationDataset = FrenchTranslationDataset(X_Test, Y_Test)

NUM_WORKERS:int = 0 #int(os.cpu_count() / 2)
NUM_BATCHES_TO_PREFETCH:int = 2

train_loader:data.DataLoader = data.DataLoader(Train_dataset, batch_size=32,
                                               shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

test_loader:data.DataLoader = data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

train_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=train_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)
test_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=test_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)

class PositionalEncoding(nn.Module):
    def __init__(self, transformer_input_feature_size, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, transformer_input_feature_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, transformer_input_feature_size, 2).float() * (-np.log(10000.0) / transformer_input_feature_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(DEVICE)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        NUM_LAYERS:int = 4
        NUM_HEADS:int = 4
        EMBEDDING_SIZE:int = 2048
        HIDDEN_SIZE:int = 2048
        DROPOUT_PROB:float = 0.2

        self.encoder_embedding:nn.Embedding = nn.Embedding(num_embeddings=NUM_ENGLISH_WORDS, embedding_dim=EMBEDDING_SIZE, padding_idx=PADDING_TOKEN)
        self.decoder_embedding:nn.Embedding = nn.Embedding(num_embeddings=NUM_FRENCH_WORDS, embedding_dim=EMBEDDING_SIZE, padding_idx=PADDING_TOKEN)

        self.positional_encoding:PositionalEncoding = PositionalEncoding(transformer_input_feature_size=EMBEDDING_SIZE)

        # * ================== nn.TransformerEncoder & nn.TransformerDecoder ==================
        self.encoder_layer:nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_SIZE, nhead=NUM_HEADS, dim_feedforward=HIDDEN_SIZE, dropout=DROPOUT_PROB,
            batch_first=True
        )
        self.encoder:nn.TransformerEncoder = nn.TransformerEncoder(self.encoder_layer, num_layers=NUM_LAYERS)

        self.decoder_layer:nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=EMBEDDING_SIZE, nhead=NUM_HEADS, dim_feedforward=HIDDEN_SIZE, dropout=DROPOUT_PROB,
            batch_first=True
        )
        self.decoder:nn.TransformerDecoder = nn.TransformerDecoder(self.decoder_layer, num_layers=NUM_LAYERS)
        # * ================================== nn.Transformer ==================================
        # self.transformer:nn.Transformer = nn.Transformer(
        #     d_model=EMBEDDING_SIZE,
        #     nhead=NUM_HEADS,
        #     num_encoder_layers=NUM_LAYERS,
        #     num_decoder_layers=NUM_LAYERS,
        #     dim_feedforward=HIDDEN_SIZE,
        #     dropout=DROPOUT_PROB,
        #     batch_first=True
        # )
        # * ====================================================================================

        self.fc_out:nn.Linear = nn.Linear(in_features=EMBEDDING_SIZE, out_features=NUM_FRENCH_WORDS)
        self.dropout:nn.Dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, X, Y):
        X_padding_mask:torch.Tensor = (X == PADDING_TOKEN)
        Y_lookahead_mask:torch.Tensor = torch.triu(torch.ones(Y.shape[1], Y.shape[1], device=DEVICE), diagonal=1).bool()
        Y_padding_mask:torch.Tensor = (Y == PADDING_TOKEN)

        X_embedded:torch.Tensor = self.dropout(self.positional_encoding(self.encoder_embedding(X)))
        Y_embedded:torch.Tensor = self.dropout(self.positional_encoding(self.decoder_embedding(Y)))

        # * ================== nn.TransformerEncoder & nn.TransformerDecoder ==================
        encoder_output:torch.Tensor = self.encoder(src=X_embedded, src_key_padding_mask=X_padding_mask)
        
        decoder_output:torch.Tensor = self.decoder(
            tgt=Y_embedded, memory=encoder_output, tgt_mask=Y_lookahead_mask, tgt_key_padding_mask=Y_padding_mask, memory_key_padding_mask=X_padding_mask
        )
        return self.fc_out(decoder_output)
        # * ================================== nn.Transformer ==================================
        # output:torch.Tensor = self.transformer(
        #     src=X_embedded, 
        #     tgt=Y_embedded, 
        #     src_key_padding_mask=X_padding_mask,
        #     tgt_key_padding_mask=Y_padding_mask,
        #     tgt_mask=Y_lookahead_mask,
        #     memory_key_padding_mask=X_padding_mask
        # )

        # return self.fc_out(output)
        # * ====================================================================================

# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))

# ========== Model Parameters ==========
model:TransformerModel = TransformerModel().to(DEVICE)
total_params = sum([p.numel() for p in model.parameters()])
print(f"Total Num Params in loaded model: {total_params:,}")

# Calculate MACs (Multiply-Accumulate Operations)
# Create sample inputs for profiling
firstBatch = next(iter(train_loader))
sample_X, sample_Y = firstBatch
sample_X, sample_Y = sample_X.to(DEVICE), sample_Y.to(DEVICE)

# Profile the model
macs = profile_macs(model, (sample_X, sample_Y))
print(f"Computational complexity: {macs:,} MACs")
print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")
# %%
# ================================================ Shape Testing ================================================
# X:torch.Tensor = torch.rand(size=(32, 32, 32, 3)).permute(0, 3, 1, 2)

# X:torch.Tensor = torch.tensor([[2, 1, 3, 4, 5, 2, 8, 1, 2, 0],
#                                [2, 1, 3, 4, 5, 2, 8, 1, 2, 0],
#                                [2, 1, 3, 4, 5, 2, 8, 1, 2, 0]])

# print(X.shape)
# print(model(X.to(DEVICE)))

# embed_ENG:nn.Embedding = nn.Embedding(num_embeddings=NUM_ENGLISH_WORDS, embedding_dim=128, padding_idx=PADDING_TOKEN)
# embed_FR:nn.Embedding = nn.Embedding(num_embeddings=NUM_FRENCH_WORDS, embedding_dim=128, padding_idx=PADDING_TOKEN)
# encoder:nn.GRU = nn.GRU(input_size=128, hidden_size=128,
#                     batch_first=True, dropout=DROPOUT_PROB, num_layers=2)

# decoder:nn.GRU = nn.GRU(input_size=128, hidden_size=128,
#                     batch_first=True, dropout=DROPOUT_PROB, num_layers=2)

# linear:nn.Linear = nn.Linear(in_features=128, out_features=NUM_FRENCH_WORDS)

firstBatch = next(iter(test_loader))
X, Y = firstBatch
X, Y = X.to(DEVICE), Y.to(DEVICE)

# Test Shape
# with torch.inference_mode():
#     out = model(X, Y)
    
# Get a single test sequence
IDX:int = 4
single_X = X[IDX].unsqueeze(0)  # Take the first sequence, keeping batch dimension
single_Y = Y[IDX].unsqueeze(0)  # Take the first target sequence for comparison

# Get model prediction
with torch.inference_mode():
    output = model(single_X, single_Y)
    
# Get the predicted token indices
predicted_indices = torch.argmax(output, dim=-1)[0]  # Remove batch dimension

# Convert the input sequence to English words
input_sequence = []
for idx in single_X[0]:
    if idx.item() != PADDING_TOKEN:
        input_sequence.append(ix_to_english_word.get(idx.item(), "<UNK>"))

# Convert the predicted sequence to French words
predicted_sequence = []
for idx in predicted_indices:
    if idx.item() != PADDING_TOKEN:
        predicted_sequence.append(ix_to_french_word.get(idx.item(), "<UNK>"))

# Convert the expected sequence to French words
expected_sequence = []
for idx in single_Y[0]:
    if idx.item() != PADDING_TOKEN:
        expected_sequence.append(ix_to_french_word.get(idx.item(), "<UNK>"))

# Print the results
print("\nInput (English):", " ".join(input_sequence))
print("\nPredicted (French):", " ".join(predicted_sequence))
print("\nExpected (French):", " ".join(expected_sequence))

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

Loss_Function:nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
# Optimizer_Function:torch.optim.Adam = torch.optim.Adam(params=model.parameters())
# Optimizer_Function:torch.optim.SGD = torch.optim.SGD(params=model.parameters(),
#                                                      lr=0.0001)
Optimizer_Function:torch.optim.Adam = torch.optim.Adam(
    params=model.parameters(),
    lr=0.00001,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-5
)

EPOCHS:int = -1
epochIterator:int = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

bestTestAccuracy:float = 0

MINIMUM_TEST_ACCURACY:int = 84

TOTAL_TRAIN_TOKENS = sum((Y_train_batch != PADDING_TOKEN).sum().item() for _, Y_train_batch in train_loader)
TOTAL_TEST_TOKENS = sum((Y_test_batch != PADDING_TOKEN).sum().item() for _, Y_test_batch in test_loader)

while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + linearOffset(input=testEpochAccuracy, offset=3, target=99) or bestTestAccuracy < MINIMUM_TEST_ACCURACY):
    startTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch, Y_train_batch in train_prefetcher:
        X_train_batch:torch.Tensor = X_train_batch.to(DEVICE, non_blocking=True)
        Y_train_batch:torch.Tensor = Y_train_batch.to(DEVICE, non_blocking=True)
        
        Y_train_pred_logits:torch.Tensor = model(X_train_batch, Y_train_batch)
        
        trainBatchLoss = Loss_Function(Y_train_pred_logits.permute(0, 2, 1), Y_train_batch.type(torch.int64))
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        padding_mask = (Y_train_batch != PADDING_TOKEN)
        numCorrectInEpoch += torch.eq(Y_train_pred_logits.argmax(dim=2), Y_train_batch)[padding_mask].sum().item()
        totalTrainLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        trainEpochAverageBatchLoss:float = totalTrainLossInEpoch/len(train_loader)
        avgTrainBatchLossPerEpoch += [trainEpochAverageBatchLoss]
        
        trainEpochAccuracy:float = numCorrectInEpoch/TOTAL_TRAIN_TOKENS * 100 # accuracy is calculated per item in a batch instead of per batch
        trainAccuracyPerEpoch += [trainEpochAccuracy]
        
        numCorrectInEpoch:int = 0
        totalTestLossInEpoch:float = 0
        for X_test_batch, Y_test_batch in test_prefetcher:
            X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
            Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
        
            Y_test_pred_logits:torch.Tensor = model(X_test_batch, Y_test_batch)
        
            testBatchLoss = Loss_Function(Y_test_pred_logits.permute(0, 2, 1), Y_test_batch.type(torch.int64))
    
            padding_mask = (Y_test_batch != PADDING_TOKEN)
            numCorrectInEpoch += torch.eq(Y_test_pred_logits.argmax(dim=2), Y_test_batch)[padding_mask].sum().item()
            
            totalTestLossInEpoch += testBatchLoss
        
        testEpochAverageBatchLoss:float = totalTestLossInEpoch/len(test_loader)
        avgTestBatchLossPerEpoch += [testEpochAverageBatchLoss]
        
        testEpochAccuracy:float = numCorrectInEpoch/TOTAL_TEST_TOKENS * 100
        testAccuracyPerEpoch += [testEpochAccuracy]
    
        epochTime:float = time.time() - startTime
        estRemainingTime:float = (EPOCHS - epochIterator - 1)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {trainEpochAverageBatchLoss:.5f}, train accuracy: {trainEpochAccuracy:.2f}% \t| test loss: {testEpochAverageBatchLoss:.5f}, test accuracy: {testEpochAccuracy:.2f}% \t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        if testEpochAccuracy > MINIMUM_TEST_ACCURACY and testEpochAccuracy > bestTestAccuracy: 
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
# %%