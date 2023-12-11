from dataLoader import dataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import numpy as np
import os
from joblib import dump
from joblib import load

class lieDetector:
    def __init__(self):
        # Extract features and labels from training, test, and validation sets

        # value_valid, labels_valid = self.extract(data_loader.packedValid)

        modelFile = 'trained-Model.joblib'
        if os.path.exists(modelFile):
            # Load the model
            self.vectorizer = load('vectorizer.joblib')
            self.model = load(modelFile)
        else:
            # Train and save the model
            self.vectorizer = TfidfVectorizer()
            self.dl = dataLoader(True)
            value_train = self.vectorizer.fit_transform(self.dl.packedTrain)
            value_test = self.vectorizer.transform(self.dl.packedTest)
            self.model = RandomForestClassifier()
            print('Data loaded')
            print('Extraction complete')
            print("Training Model")
            #self.model.fit(value_train, labels_train)
            self.model.fit(value_train, np.array(self.dl.trainLabels))
            # Predict and evaluate
            # predictions = self.model.predict(value_test)
            # ... evaluate predictions ...
            # Calculate accuracy on the test set
            accuracy = self.model.score(value_test, np.array(self.dl.testLabels))
            print("Accuracy on test set:", accuracy)
            dump(self.model, 'trained-Model.joblib')
            dump(self.vectorizer, 'vectorizer.joblib')


    def predict(self, inputs):
        processed_input = ' '.join(
            word.lower() for word in inputs.split() if word.lower() not in self.dl.stop_words)

        # Transform the processed_input using the vectorizer

        featStatement = self.vectorizer.transform([processed_input])

        # Convert the result of transform to a dense array
        featVecs = featStatement.toarray()

        # Use the dense array for prediction
        print("Predicting for: " + inputs)
        prediction = self.model.predict(featVecs)
        print(prediction[0])
        return "False" if prediction[0] == 0 else "Completely True"




