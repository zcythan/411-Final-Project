from dataLoader import dataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from joblib import dump
from joblib import load
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class lieDetector:
    def __init__(self):
        # Extract features and labels from training, test, and validation sets
        # value_valid, labels_valid = self.extract(data_loader.packedValid)
        self.stopWords = set(stopwords.words('english'))
        modelFile = 'trained-Model.joblib'
        vectFile = 'vectorizer.joblib'
        if os.path.exists(modelFile) and os.path.exists(vectFile):
            # Load the model
            self.vectorizer = load(vectFile)
            self.model = load(modelFile)
        else:
            # Train and save the model
            print("AI agent or Vectorizer Files are missing, creating them now...")
            print("This may take up to 5 minutes.")
            self.vectorizer = TfidfVectorizer()
            self.dl = dataLoader(True)
            value_train = self.vectorizer.fit_transform(self.dl.packedTrain)
            value_test = self.vectorizer.transform(self.dl.packedTest)
            self.model = RandomForestClassifier()
            print('Data loaded')
            print('Extraction complete')
            print("Training Model")
            self.model.fit(value_train, np.array(self.dl.trainLabels))
            accuracy = self.model.score(value_test, np.array(self.dl.testLabels))
            print("Accuracy on test set:", accuracy)
            dump(self.model, 'trained-Model.joblib')
            dump(self.vectorizer, 'vectorizer.joblib')


    def predict(self, inputs):
        processed_input = ' '.join(
            word.lower() for word in inputs.split() if word.lower() not in self.stopWords)

        # Transform the processed_input using the vectorizer
        featStatement = self.vectorizer.transform([processed_input])

        # Convert the result of transform to a dense array
        featVecs = featStatement.toarray()

        # Use the dense array for prediction
        print("Predicting for: " + inputs)
        prediction = self.model.predict(featVecs)
        print(prediction[0])
        return "This statement is likely falsified." if prediction[0] == 0 else "This statement is likely truthful."




