from dataLoader import dataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

class lieDetector:
    def __init__(self):
        data_loader = dataLoader()
        print('loaded')

        value_train, labels_train = self.extract(data_loader.packedTrain)
        value_test, labels_test = self.extract(data_loader.packedTest)
        value_valid, labels_valid= self.extract(data_loader.packedValid)

        value_train = np.asarray(value_train)
        value_test = np.asarray(value_test)

        print('extraction complete')

        nb = MultinomialNB()
        nb.fit(value_train, labels_train.values.ravel())  # Use values.ravel() to avoid shape mismatch

        # Predict on the test set
        predictions = nb.predict(value_test)

        # Evaluate the accuracy or other metrics
        accuracy = nb.score(value_test, labels_test)
        print("Accuracy of Naive Bayes Classifier:", accuracy * 100)

        #X_train, X_test, Y_train, Y_test = train_test_split(tfidf_matrix,y_df, random_state=2)
        #nb = MultinomialNB()
        #nb.fit(X_train, Y_train)
        #Accuracy_NB = nb.score(X_test, Y_test)
        #print(Accuracy_NB*100)

    def extract(self, mode):
        text_data = []
        numerical_data = []
        labels = []

        for dictionary in mode:
            if 'label' in dictionary: 
                labels.append(dictionary['label'])  
                del dictionary['label'] 

        df = pd.DataFrame(mode)
        y_df = pd.DataFrame(labels);

        text_columns = ["statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "context"]
        numerical_columns = df.columns.difference(text_columns)

        text_data = df[text_columns].agg(' '.join, axis=1)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_data)

        numerical_data_df = df[numerical_columns].fillna(0)  # Replace NaNs with 0 or any other suitable value
        numerical_data_array = numerical_data_df.to_numpy()

        tfidf_dense = tfidf_matrix.todense()
        combined_features = np.hstack((tfidf_dense, numerical_data_array))

        print('data extracted')
        return combined_features, y_df

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
