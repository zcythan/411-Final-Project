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

        binmodelFile = 'trained-ModelBin.joblib'
        sixmodelFile = 'trained-ModelSix.joblib'
        binVectFile = 'vectorizerBin.joblib'
        sixVectFile = 'vectorizerSix.joblib'
        if os.path.exists(binmodelFile) and os.path.exists(binVectFile) and os.path.exists(sixmodelFile) and os.path.exists(binmodelFile):
            # Load the model
            self.binVectorizer = load(binVectFile)
            self.binModel = load(binmodelFile)
            self.sixModel = load(sixmodelFile)
            self.sixVectorizer = load(sixVectFile)
        else:
            # Train and save the model
            print("AI agent or Vectorizer Files are missing, creating them now...")
            print("This may take up to 5 minutes.")
            self.binVectorizer = TfidfVectorizer()
            self.sixVectorizer = TfidfVectorizer()
            # Change this false to run 6 label classification.
            self.bindl = dataLoader(True)
            self.sixdl = dataLoader(False)

            binValue_train = self.binVectorizer.fit_transform(
                self.bindl.packedTrain)
            binValue_test = self.binVectorizer.transform(self.bindl.packedTest)
            sixValue_train = self.sixVectorizer.fit_transform(
                self.sixdl.packedTrain)
            sixValue_test = self.sixVectorizer.transform(self.sixdl.packedTest)
            self.binModel = RandomForestClassifier()
            self.sixModel = RandomForestClassifier()
            print('Data loaded')
            print('Extraction complete')
            print("Training Model")
            self.binModel.fit(binValue_train, np.array(self.bindl.trainLabels))
            self.sixModel.fit(sixValue_train, np.array(self.sixdl.trainLabels))
            accuracy = self.binModel.score(
                binValue_test, np.array(self.bindl.testLabels))
            print("Accuracy on binary test set:", accuracy)
            accuracy = self.sixModel.score(
                sixValue_test, np.array(self.sixdl.testLabels))
            print("Accuracy on six-way test set:", accuracy)
            dump(self.binModel, binmodelFile)
            dump(self.sixModel, sixmodelFile)
            dump(self.binVectorizer, binVectFile)
            dump(self.sixVectorizer, sixVectFile)

    def predict(self, inputs, binary):
        processed_input = ' '.join(
            word.lower() for word in inputs.split() if word.lower() not in self.stopWords)

        if binary:
            featStatement = self.binVectorizer.transform([processed_input])
            featVecs = featStatement.toarray()

            print("Predicting for: " + inputs)
            prediction = self.binModel.predict(featVecs)
            print(prediction[0])
            return prediction[0]
        else:
            featStatement = self.sixVectorizer.transform([processed_input])

            featVecs = featStatement.toarray()

            print("Predicting for: " + inputs)
            prediction = self.sixModel.predict(featVecs)
            print(prediction[0])
            output_list = ["False", "Half True", "Mostly True",
                           "Completely True", "Barely True", "Extremely False"]
            return output_list[prediction[0]]
