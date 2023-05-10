#!/usr/bin/env python
# coding: utf-8


# import packages and settings

import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import nltk
from nltk import word_tokenize    
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from collections import Counter
from torch.utils.data import Dataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# read data for command line
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--train_csv_file', type=str, required=True, help='Path to the training CSV file')
parser.add_argument('--valid_csv_file', type=str, required=True, help='Path to the validation CSV file')
parser.add_argument('--test_csv_file', type=str, required=True, help='Path to the test CSV file')
parser.add_argument('--output_csv_file', type=str, required=True, help='Path to the output CSV file')
args = parser.parse_args()

train_data = pd.read_csv(args.train_csv_file)
valid_data = pd.read_csv(args.valid_csv_file)
test_data = pd.read_csv(args.test_csv_file)





# # read data

# train_data = pd.read_csv('TaskA-TrainingSet.csv')
# valid_data = pd.read_csv('TaskA-ValidationSet.csv')

# Extract the dialogues and labels from the training data
dialogues = train_data['dialogue'].tolist()
labels = train_data['section_header'].tolist()

# assign train/ valid

X_train = train_data['dialogue']
y_train = train_data['section_header']

X_valid = valid_data['dialogue']
y_valid = valid_data['section_header']


# clean/ preprocess


stop_words = set(stopwords.words('english'))

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")



def clean_data(text):
    
    # remove "Doctor:" and "Patient:" labels and timestamps
    text = re.sub(r"(Doctor|Patient|Guest_family):|\d{1,2}[:.]\d{1,2}\s?(AM|PM|am|pm)?", "", text)
    
    # Lowercase the text
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]+", "", text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a single string
    text = " ".join(words)

    return text



# preprocess the train data

X_train = X_train.apply(clean_data)
X_valid = X_valid.apply(clean_data)


# + oversampler and run Clinical BERT - best result - F1: 0.75

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=20)

class Dialogue2TopicDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        # Encode
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }

# convert  

max_length = 307  

# Prepare label encoding
label_to_id = {label: idx for idx, label in enumerate(set(y_train))}
id_to_label = {idx: label for label, idx in label_to_id.items()}


# Oversampling 
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(
    np.array(X_train).reshape(-1, 1), y_train
)
X_train_resampled = X_train_resampled.ravel()

# print("Before oversampling:", Counter(y_train))
# print("After oversampling:", Counter(y_train_resampled))

# Encode label
y_train_ids = [label_to_id[label] for label in y_train_resampled]  
y_valid_ids = [label_to_id[label] for label in y_valid]

# convert dataset
train_dataset = Dialogue2TopicDataset(X_train_resampled, y_train_ids, tokenizer, max_length)  
valid_dataset = Dialogue2TopicDataset(X_valid, y_valid_ids, tokenizer, max_length)

# Define the perameters

training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    learning_rate=2e-5,  
    weight_decay=0.01,  
    save_strategy="no",
)

# metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#  train model 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# evaluation

evaluation_results = trainer.evaluate()
print(evaluation_results)

# report

predictions_output = trainer.predict(valid_dataset)
predicted_labels = predictions_output.predictions.argmax(-1)

y_valid_pred = [id_to_label[label_id] for label_id in predicted_labels]
y_valid_true = [id_to_label[label_id] for label_id in y_valid_ids]

report = classification_report(y_valid_true, y_valid_pred)
print(report)


# predict test labels



# # test data
# test_data = pd.read_csv("taskA_testset4participants_headers_inputConversations.csv")

# Assign test data
X_test = test_data['dialogue']


# Preprocess test data
X_test = X_test.apply(clean_data)


# Create test dataset (without labels)
class Dialogue2TopicTestDataset(Dataset):
    def __init__(self, X, tokenizer, max_length):
        self.X = X
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]

        # Encode
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0]
        }


# convert test data

test_dataset = Dialogue2TopicTestDataset(X_test, tokenizer, max_length)
test_dataset


# Predict test data
test_predictions_output = trainer.predict(test_dataset)
test_predicted_labels = test_predictions_output.predictions.argmax(-1)

# Convert label ids back to original labels
y_test_pred = [id_to_label[label_id] for label_id in test_predicted_labels]
y_test_pred


# Create the df 
output_df = pd.DataFrame({'TestID': test_data['ID'], 'SystemOutput': y_test_pred})

# CSV file
# output_df.to_csv('taskA_StellElla_Stars_run1_mediqaSum.csv', index=False)

output_df.to_csv(args.output_csv_file, index=False)


