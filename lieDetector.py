from dataLoader import dataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


class lieDetector:
    def __init__(self):
        data_loader = dataLoader()
        print()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        # Initialize dataLoader
        self.train_data = self.preprocess(data_loader.packedTrain)
        self.test_data = self.preprocess(data_loader.packedTest)
        self.valid_data = self.preprocess(data_loader.packedValid)

    def preprocess(self, data):
        # Convert the data to the format required by BERT
        input_ids = []
        attention_masks = []
        labels = []

        for item in data:
            encoded_dict = self.tokenizer.encode_plus(
                item['statement'],  # Input text
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attention masks
                return_tensors='pt',  # Return pytorch tensors
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(item['label'])  # Assuming 'label' is the key for labels in your dataset

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return TensorDataset(input_ids, attention_masks, labels)

    def train(self, epochs=4, batch_size=32):
        # Create DataLoader for training data
        train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                scheduler.step()

        print("Training complete")

    def predict(self, text):
        # Prediction on text
        self.model.eval()
        with torch.no_grad():
            encoded_dict = self.tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=64, pad_to_max_length=True, return_attention_mask=True,
                return_tensors='pt')
            input_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
            outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = outputs[0]
            index = logits.argmax()

        return index.item()


''' 
def __init__(self):
    self.__var = 0
    data = dataLoader()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def load(self):
    #code to load in model here
    placeholder = 0

def predict(self, text): # add in a parameter for the text to be predicted
    #Does prediction based on data
    placeholder = 0
    return placeholder
    '''
