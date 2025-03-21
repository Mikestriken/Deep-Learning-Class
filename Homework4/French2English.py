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
# SOS_TOKEN = 2
NUM_ADDED_TOKENS = 2

NUM_ENGLISH_WORDS:int = len(english_words) + NUM_ADDED_TOKENS
NUM_FRENCH_WORDS:int = len(french_words) + NUM_ADDED_TOKENS

english_word_to_ix = {"PAD": PADDING_TOKEN, "EOS": EOS_TOKEN, **{ch: i + NUM_ADDED_TOKENS for i, ch in enumerate(english_words)}} 
ix_to_english_word = {i: ch for i, ch in english_word_to_ix.items()}

french_word_to_ix = {"PAD": PADDING_TOKEN, "EOS": EOS_TOKEN, **{ch: i + NUM_ADDED_TOKENS for i, ch in enumerate(french_words)}} 
ix_to_french_word = {i: ch for i, ch in french_word_to_ix.items()}

# ========== Train Set ==========
X_Train = [[french_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Train_dataset[:,1]]
Y_Train = [[english_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Train_dataset[:,0]]

X_Train = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in X_Train], batch_first=True, padding_value=PADDING_TOKEN)
Y_Train = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in Y_Train], batch_first=True, padding_value=PADDING_TOKEN)

# ========== Test Set ==========
X_Test = [[french_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Test_dataset[:,1]]
Y_Test = [[english_word_to_ix[word] for word in sentence.split()] + [EOS_TOKEN] for sentence in Test_dataset[:,0]]

X_Test = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in X_Test], batch_first=True, padding_value=PADDING_TOKEN)
Y_Test = pad_sequence([torch.tensor(sentence, dtype=torch.long) for sentence in Y_Test], batch_first=True, padding_value=PADDING_TOKEN)

class FrenchTranslationDataset(data.Dataset):
    def __init__(self, features, labels, padding_token = PADDING_TOKEN):
        self.features = features
        self.labels = labels
        
        # elements in features are sequences, sequences have different lengths that are padded with padding_token (unique token)
        self.features_Sequence_Lengths = [torch.sum(feature != padding_token).item() for feature in features]
        # self.labels_Sequence_Lengths = [torch.sum(label != padding_token).item() for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.features_Sequence_Lengths[idx]#, self.labels_Sequence_Lengths[idx]

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

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_prob, padding_token = PADDING_TOKEN):
        super().__init__()
        self.padding_token:int = padding_token
        
        self.wordEmbeddingLayer:nn.Embedding = nn.Embedding(num_embeddings=NUM_FRENCH_WORDS, embedding_dim=embedding_size, padding_idx=padding_token)
        
        self.rnn:nn.GRU = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                                 batch_first=True, dropout=dropout_prob if num_layers > 1 else 0.0, num_layers=num_layers)
        
    
    def forward(self, X:torch.Tensor, Input_Hidden_State = None, X_sequence_lengths:torch.Tensor = None):
        # X.shape = (batch_size, seq_length, input_size)
        # hidden.shape = (num_layers, batch_size, hidden_size)
        # rnn_out.shape = (batch_size, seq_length, hidden_size)
        
        X_Embedded = self.wordEmbeddingLayer(X)
        
        output:torch.Tensor
        Hidden_State:torch.Tensor
        if X.dim() > 1:
            X_Embedded_Packed = pack_padded_sequence(X_Embedded, X_sequence_lengths.cpu(), batch_first=True, enforce_sorted=True)
            
            Output_Packed, Hidden_State = self.rnn(X_Embedded_Packed, Input_Hidden_State)
            
            output, _ = pad_packed_sequence(Output_Packed, batch_first=True, padding_value=self.padding_token)
        
        else:
            # Input_Hidden_State = Input_Hidden_State.unsqueeze(dim=0) if Input_Hidden_State is not None else Input_Hidden_State
            
            output, Hidden_State = self.rnn(X_Embedded.unsqueeze(dim=0), Input_Hidden_State) # (batch_size=1, seq_length, input_size)
            
        return output, Hidden_State
        
class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_prob, padding_token = PADDING_TOKEN):
        super().__init__()
        
        self.padding_token:int = padding_token
        
        self.wordEmbeddingLayer:nn.Embedding = nn.Embedding(num_embeddings=NUM_ENGLISH_WORDS, embedding_dim=embedding_size, padding_idx=padding_token)
        
        self.rnn:nn.GRU = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                                 batch_first=True, dropout=dropout_prob if num_layers > 1 else 0.0, num_layers=num_layers)
        
        self.outputDenseLayer = nn.Sequential(
            
            nn.Linear(in_features=hidden_size, out_features=NUM_ENGLISH_WORDS),
        )
    
    def forward(self, X, Input_Hidden_State = None):
        # x.shape = (batch_size, seq_length, input_size)
        # hidden.shape = (num_layers, batch_size, hidden_size)
        # rnn_out.shape = (batch_size, seq_length, hidden_size)
        
        X_Embedded = self.wordEmbeddingLayer(X)
        
        output:torch.Tensor
        Hidden_State:torch.Tensor
        if X.dim() > 1:
            output, Hidden_State = self.rnn(X_Embedded, Input_Hidden_State)
        else:
            # Input_Hidden_State = Input_Hidden_State.unsqueeze(dim=0) if Input_Hidden_State is not None else Input_Hidden_State
            
            output, Hidden_State = self.rnn(X_Embedded.unsqueeze(dim=0), Input_Hidden_State)
            
        return self.outputDenseLayer(output), Hidden_State

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        
        NUM_RNN_LAYERS:int = 2
        EMBEDDING_SIZE:int = 2048
        HIDDEN_SIZE:int = 2048
        DROPOUT_PROB:float = 0.8
        
        self.encoder:Encoder = Encoder(embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_RNN_LAYERS, dropout_prob=DROPOUT_PROB).to(DEVICE)
        self.decoder:Decoder = Decoder(embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_RNN_LAYERS, dropout_prob=DROPOUT_PROB).to(DEVICE)
        
    def forward(self, X:torch.Tensor, Y_Ground_Truth:torch.Tensor = None, X_sequence_lengths:torch.Tensor = None):
        
        # * Were labels provided (or are we inferencing?)
        Teacher_Forcing_Enabled:bool = True
        if Y_Ground_Truth is None: 
            assert torch.is_inference_mode_enabled(), "No Y_Ground_Truth provided. If inferencing use `with torch.inference_mode()`"
            Teacher_Forcing_Enabled:bool = False
        
        # Determine if input is batched
        X_isBatched:bool = False
        if X.dim() > 1:
            assert X_sequence_lengths is not None, "If multiple batches are being inserted, provide X_sequence_lengths"
            assert Teacher_Forcing_Enabled, "If multiple (batched) inputs are provided, a Y_Ground_Truth must be provided for each input"
            X_isBatched:bool = True
        else:
            assert X_sequence_lengths is None, "If multiple batches are being inserted, provide X_sequence_lengths"
            
        # * Pass input through encoder
        encoder_last_hidden_state:torch.Tensor
        sortedIndices:torch.LongTensor
        unsortedIndices:torch.LongTensor
        if X_isBatched:
            # Format Input
            X_sequence_lengths_sorted, sortedIndices = X_sequence_lengths.sort(descending=True)
            _, unsortedIndices = sortedIndices.sort()
            
            X = X[sortedIndices]
            
            # Pass input through encoder
            _, encoder_last_hidden_state = self.encoder(X=X, Input_Hidden_State=None, X_sequence_lengths=X_sequence_lengths_sorted)
        
        else:
            encoder_last_hidden_state = None # Initial hidden state is nothing
            
            for sequence_index in range(X.shape[0]):
                # X.shape => (sequence_length)
                
                input = torch.tensor([X[sequence_index]], dtype=torch.long, device=DEVICE)
                
                _, encoder_last_hidden_state = self.encoder(X=input, Input_Hidden_State=encoder_last_hidden_state)
        
        # * Pass through decoder
        if X_isBatched and Teacher_Forcing_Enabled:
            # rnn_out.shape = (batch_size, seq_length=1, hidden_size)
            
            # Decode first input with EOS and encoder hidden state
            batch_size = X.shape[0]
            first_input = torch.full(size=(batch_size, 1), fill_value=EOS_TOKEN).to(DEVICE)
            decoder_output_sorted, last_decoder_hidden_state = self.decoder(X=first_input, Input_Hidden_State=encoder_last_hidden_state)
            
            decoded_outputs_sorted:list = [decoder_output_sorted]
            
            Y_Ground_Truth_Sorted = Y_Ground_Truth[sortedIndices]
            for sequence_index in range(Y_Ground_Truth.shape[1] - 1): # Note: -1 to remove EOS token at end of sequence
                # Y_Ground_Truth.shape => (batch_size, sequence_length)
                # Y_Ground_Truth[:, sequence_index].unsqueeze(dim=1).shape => (batch_size, sequence_length = 1)
                sequence_n_words = Y_Ground_Truth_Sorted[:, sequence_index].unsqueeze(dim=1)
                
                decoder_output_sorted, last_decoder_hidden_state = self.decoder(X=sequence_n_words, Input_Hidden_State=last_decoder_hidden_state)
                decoded_outputs_sorted.append(decoder_output_sorted)
            
            decoded_outputs_sorted:torch.Tensor = torch.stack(decoded_outputs_sorted, dim=1).squeeze() # convert list to tensor
            decoded_outputs = decoded_outputs_sorted[unsortedIndices]
            
            return decoded_outputs
            
        elif X_isBatched and not Teacher_Forcing_Enabled:
            "NOT SUPPORTED"
            
        
        elif not X_isBatched and Teacher_Forcing_Enabled:
            # rnn_out.shape = (batch_size=0, seq_length=0, hidden_size)
        
            # Decode first input with EOS and encoder hidden state
            first_input = torch.tensor([EOS_TOKEN], dtype=torch.long, device=DEVICE)
            decoder_output, last_decoder_hidden_state = self.decoder(X=first_input, Input_Hidden_State=encoder_last_hidden_state)
            
            decoded_outputs:list = [decoder_output]
            
            for sequence_index in range(Y_Ground_Truth.shape[0] - 1): # Note: -1 to remove EOS token at end of sequence
                # Y_Ground_Truth.shape => (sequence_length)
                sequence_n_word = torch.tensor([Y_Ground_Truth[sequence_index]], dtype=torch.long, device=DEVICE)
                
                decoder_output, last_decoder_hidden_state = self.decoder(X=sequence_n_word, Input_Hidden_State=last_decoder_hidden_state)
                decoded_outputs.append(decoder_output)
            
            decoded_outputs:torch.Tensor = torch.stack(decoded_outputs, dim=1).squeeze() # convert list to tensor
            
            return decoded_outputs
                
        
        elif not X_isBatched and not Teacher_Forcing_Enabled:
            # rnn_out.shape = (batch_size=0, seq_length=0, hidden_size)
            
            # first_input = torch.full(size=(1,), fill_value=EOS_TOKEN).to(DEVICE)
            first_input = torch.tensor([EOS_TOKEN], dtype=torch.long, device=DEVICE)
            decoder_output, last_decoder_hidden_state = self.decoder(first_input, encoder_last_hidden_state)
            
            decoded_outputs:list = [decoder_output]
            
            while decoder_output.squeeze().argmax().item() != EOS_TOKEN:
                decoder_output, last_decoder_hidden_state = self.decoder(X=decoder_output.argmax().unsqueeze(dim=0), Input_Hidden_State=last_decoder_hidden_state)
                decoded_outputs.append(decoder_output)
                
            decoded_outputs:torch.Tensor = torch.stack(decoded_outputs, dim=1).squeeze() # convert list to tensor
            
            return decoded_outputs
# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))

# ========== Model Parameters ==========

model:Seq2Seq = Seq2Seq().to(DEVICE)
print(f"Total Num Params in loaded model: {sum([p.numel() for p in model.parameters()])}")
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

firstBatch = next(iter(train_loader))
X, Y, X_sequence_lengths = firstBatch
X, Y, X_sequence_lengths = X.to(DEVICE), Y.to(DEVICE), X_sequence_lengths.to(DEVICE)

with torch.inference_mode():
    out = model(X[1])

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
Optimizer_Function:torch.optim.Adam = torch.optim.Adam(params=model.parameters())#,
                                                    #  lr=0.15) #0.15

EPOCHS:int = 50
epochIterator:int = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

bestTestAccuracy:float = 0

MINIMUM_TEST_ACCURACY:int = 40

TOTAL_TRAIN_TOKENS = sum((Y_train_batch != PADDING_TOKEN).sum().item() for _, Y_train_batch, _ in train_loader)
TOTAL_TEST_TOKENS = sum((Y_test_batch != PADDING_TOKEN).sum().item() for _, Y_test_batch, _ in test_loader)

while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + linearOffset(input=testEpochAccuracy, offset=3, target=99) or bestTestAccuracy < MINIMUM_TEST_ACCURACY):
    startTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch, Y_train_batch, X_train_batch_sequence_lengths in train_prefetcher: #==
        X_train_batch:torch.Tensor = X_train_batch.to(DEVICE, non_blocking=True)
        Y_train_batch:torch.Tensor = Y_train_batch.to(DEVICE, non_blocking=True)
        X_train_batch_sequence_lengths:torch.Tensor = X_train_batch_sequence_lengths.to(DEVICE, non_blocking=True)
        
        Y_train_pred_logits:torch.Tensor = model(X_train_batch, Y_train_batch, X_train_batch_sequence_lengths)
        
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
        for X_test_batch, Y_test_batch, X_test_batch_sequence_lengths in test_prefetcher: #==
            X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
            Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
            X_test_batch_sequence_lengths:torch.Tensor = X_test_batch_sequence_lengths.to(DEVICE, non_blocking=True)
        
            Y_test_pred_logits:torch.Tensor = model(X_test_batch, Y_test_batch, X_test_batch_sequence_lengths)
            Y_test_pred:torch.Tensor = Y_test_pred_logits.argmax(dim=2)
        
            testBatchLoss = Loss_Function(Y_test_pred_logits.permute(0, 2, 1), Y_test_batch.type(torch.int64))
    
            padding_mask = (Y_test_batch != PADDING_TOKEN)
            numCorrectInEpoch += torch.eq(Y_test_pred, Y_test_batch).sum().item()
            
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