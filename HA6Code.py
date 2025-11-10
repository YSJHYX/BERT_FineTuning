'''
Developer Name: HONG Yuxiang

Developer Email: yhongbb@connect.ust.hk
'''
import os
os.environ["TORCH_HOME"]="./torch"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertForSequenceClassification, BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load data
df = pd.read_csv("IMDB Dataset.csv")
# print(df.head())

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df[['review', 'label']]

# print("Number of sentences: ", df.shape[0])
# Number of sentences:  50000

# print('Label 0 Num: ', (df['label']==0).sum())
# print('Label 1 Num: ', (df['label']==1).sum())
'''
Label 0 Num:  25000
Label 1 Num:  25000
'''

# check class distribution
# print(df['label'].value_counts(normalize=True))
'''
label
1    0.5
0    0.5
Name: proportion, dtype: float64
'''

# shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# print("Training dataset size: ", len(train_df))
# print("Testing dataset size: ", len(test_df))
'''
Training dataset size:  40000
Testing dataset size:  10000
'''

# import BERT model and BERT tokenizer
# import pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
# load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# define a function to encode the text
def encode_texts(texts, max_length=128):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Define the Dataset class
class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)
    
class CustomBertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.bert=BertModel.from_pretrained(bert_model_name)
        self.dropout=nn.Dropout(0.1)
        self.classifier=nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state 

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        token_embedding = sum_embeddings / sum_mask

        token_embedding = self.dropout(token_embedding)
        logits = self.classifier(token_embedding)  
        return logits
    
# Prepare the datasets
MAX_LENGTH = 128
train_encodings = encode_texts(train_df['review'], max_length=MAX_LENGTH)
test_encodings = encode_texts(test_df['review'], max_length=MAX_LENGTH)

train_labels = torch.tensor(train_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

train_dataset = IMDBDataset(train_encodings, train_labels)
test_dataset = IMDBDataset(test_encodings, test_labels)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# define the opitimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
# loss function
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        loss = loss_fn(outputs.logits, labels)
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy

# define evaluation function
def eval_model(model, dataloader, loss_fn, device):
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = loss_fn(outputs.logits, labels)
            logits = outputs.logits

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples

    
    report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
    
    return avg_loss, accuracy, report


# EPOCHS = 3
# for epoch in range(EPOCHS):
#     print(f"===== Epoch {epoch + 1}/{EPOCHS} =====")
#     train_loss, train_acc = train_epoch(
#         model, train_loader, optimizer, loss_fn, device
#     )
#     print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

#     test_loss, test_acc, test_report = eval_model(
#         model, test_loader, loss_fn, device
#     )
#     print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
#     print("\nClassification Report:\n", test_report)


# N_TRAIN = 400   
# N_TEST = 100

# train_df = train_df.iloc[:N_TRAIN]
# test_df = test_df.iloc[:N_TEST]

def run_bert_experiment(train_df, test_df, n_train, n_test, model='classic', lr=2e-5, batch_size=4, epochs=3, max_length=128, dropout_rate=0.1, freeze_bert=False, freeze_partial=False, num_frozen_layers=6, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = train_df.iloc[:n_train]
    test_df = test_df.iloc[:n_test]

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = encode_texts(train_df['review'], max_length=max_length)
    test_encodings = encode_texts(test_df['review'], max_length=max_length)

    train_labels = torch.tensor(train_df['label'].values)
    test_labels = torch.tensor(test_df['label'].values)

    train_dataset = IMDBDataset(train_encodings, train_labels)
    test_dataset = IMDBDataset(test_encodings, test_labels)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model == 'classic':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
    elif model == 'custom':
        model = CustomBertClassifier(num_labels=2)
    model = model.to(device)

    if freeze_bert:
        for param in model.base_model.parameters():
            param.requires_grad = False

    if freeze_partial:
        for i, (name, param) in enumerate(model.base_model.named_parameters()):
            try:
                layer_num = int(name.split('.')[2]) 
                if layer_num < num_frozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            except (IndexError, ValueError):
                param.requires_grad = True


    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"===== Epoch {epoch + 1}/{epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        test_loss, test_acc, test_report = eval_model(model, test_loader, loss_fn, device)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        print("\nClassification Report:\n", test_report)


# Experiment 1: Fine-tuning BERT with default parameters
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=3, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 2: Using different amounts of data for training and testing
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=400, n_test=100, lr=2e-5, batch_size=4, epochs=3, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 3: Freezing BERT layers
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=3, freeze_bert=True, freeze_partial=False, device=None)

# Experiment 4: Varying learning rates
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=1e-5, batch_size=4, epochs=3, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 5: Varying batch sizes
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=8, epochs=3, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 6: Varying number of epochs
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=6, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 7: Freezing partial BERT layers
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=3, freeze_bert=False, freeze_partial=True, num_frozen_layers=6, device=None)

# Experiment 8: Varing the max length
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=3, max_length=256, freeze_bert=False, freeze_partial=False, device=None)

# Experiment 9: Varing dropout rate
run_bert_experiment(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, lr=2e-5, batch_size=4, epochs=3, dropout_rate=0.3, freeze_bert=False, freeze_partial=False, device=None)







# ------------------------------------------------ Special Case ------------------------------------------------ #
# Need to adjust the train and evaluation parts to use the custom model properly
def train_epoch1(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

        logits = outputs
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy




def eval_model1(model, dataloader, loss_fn, device):
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples

    
    report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
    
    return avg_loss, accuracy, report




def run_bert_experiment1(train_df, test_df, n_train, n_test, model='classic', lr=2e-5, batch_size=4, epochs=3, max_length=128, dropout_rate=0.1, freeze_bert=False, freeze_partial=False, num_frozen_layers=6, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = train_df.iloc[:n_train]
    test_df = test_df.iloc[:n_test]

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = encode_texts(train_df['review'], max_length=max_length)
    test_encodings = encode_texts(test_df['review'], max_length=max_length)

    train_labels = torch.tensor(train_df['label'].values)
    test_labels = torch.tensor(test_df['label'].values)

    train_dataset = IMDBDataset(train_encodings, train_labels)
    test_dataset = IMDBDataset(test_encodings, test_labels)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model == 'classic':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
    elif model == 'custom':
        model = CustomBertClassifier(num_labels=2)
    model = model.to(device)

    if freeze_bert:
        for param in model.base_model.parameters():
            param.requires_grad = False

    if freeze_partial:
        for i, (name, param) in enumerate(model.base_model.named_parameters()):
            try:
                layer_num = int(name.split('.')[2]) 
                if layer_num < num_frozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            except (IndexError, ValueError):
                param.requires_grad = True


    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"===== Epoch {epoch + 1}/{epochs} =====")
        train_loss, train_acc = train_epoch1(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        test_loss, test_acc, test_report = eval_model1(model, test_loader, loss_fn, device)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        print("\nClassification Report:\n", test_report)






# Experiment 10: Using the output embeddings of other tokens instead of that of the class token
run_bert_experiment1(train_df=train_df, test_df=test_df, n_train=40000, n_test=10000, model='custom', lr=2e-5, batch_size=4, epochs=3, freeze_bert=False, freeze_partial=False, device=None)
