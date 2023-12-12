from dataLoader import dataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from joblib import dump
from joblib import load
import nltk
import threading
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
        if os.path.exists(binmodelFile) and os.path.exists(binVectFile) and os.path.exists(sixmodelFile) and os.path.exists(sixVectFile):
            # Load the model
            self.binVectorizer = load(binVectFile)
            self.binModel = load(binmodelFile)
            self.sixModel = load(sixmodelFile)
            self.sixVectorizer = load(sixVectFile)
        else:
            if os.path.exists(binmodelFile):
                os.remove(binmodelFile)
            if os.path.exists(binVectFile):
                os.remove(binVectFile)
            if os.path.exists(sixmodelFile):
                os.remove(sixmodelFile)
            if os.path.exists(sixVectFile):
                os.remove(sixVectFile)
            # Train and save the model
            print("AI agent or Vectorizer Files are missing, creating them now...")
            print("This may take up to 5 minutes.")
            self.binVectorizer = TfidfVectorizer()
            self.sixVectorizer = TfidfVectorizer()
            # Change this false to run 6 label classification.
            self.bindl = None
            self.sixdl = None

            tBinDl = threading.Thread(target=self.loadBinDl)
            tSixDl = threading.Thread(target=self.loadSixDl)
            tBinDl.start()
            tSixDl.start()
            tBinDl.join()
            tSixDl.join()

            self.binValue_train = self.binVectorizer.fit_transform(self.bindl.packedTrain)
            self.binValue_test = self.binVectorizer.transform(self.bindl.packedTest)
            self.sixValue_train = self.sixVectorizer.fit_transform(self.sixdl.packedTrain)
            self.sixValue_test = self.sixVectorizer.transform(self.sixdl.packedTest)
            self.binModel = RandomForestClassifier()
            self.sixModel = RandomForestClassifier()
            print('Data loaded')
            print('Extraction complete')
            tBinModel = threading.Thread(target=self.trainBinModel())
            tSixModel = threading.Thread(target=self.trainSixModel())
            tBinModel.start()
            tSixModel.start()
            tBinModel.join()
            tSixModel.join()
            accuracy = self.binModel.score(self.binValue_test, np.array(self.bindl.testLabels))
            print("Accuracy on binary test set:", accuracy)
            accuracy = self.sixModel.score(self.binValue_test, np.array(self.sixdl.testLabels))
            print("Accuracy on six-way test set:", accuracy)
            dump(self.binModel, binmodelFile)
            dump(self.sixModel, sixmodelFile)
            dump(self.binVectorizer, binVectFile)
            dump(self.sixVectorizer, sixVectFile)


    def trainBinModel(self):
        print("Training and testing Binary Model")
        self.binModel.fit(self.binValue_train, np.array(self.bindl.trainLabels))

    def trainSixModel(self):
        print("Training and testing Six Model")
        self.sixModel.fit(self.sixValue_train, np.array(self.sixdl.trainLabels))

    def loadBinDl(self):
        self.bindl = dataLoader(True)
    def loadSixDl(self):
        self.sixdl = dataLoader(False)

    def predict(self, inputs, binary):
        processedText = ' '.join(word.lower() for word in inputs.split() if word.lower() not in self.stopWords)

        if binary:
            featStatement = self.binVectorizer.transform([processedText])
            featVecs = featStatement.toarray()

            print("Predicting for: " + inputs)
            prediction = self.binModel.predict(featVecs)
            print(prediction[0])
            return prediction[0]
        else:
            featStatement = self.sixVectorizer.transform([processedText])

            featVecs = featStatement.toarray()

            print("Predicting for: " + inputs)
            prediction = self.sixModel.predict(featVecs)
            print(prediction[0])
            output_list = ["False", "Half True", "Mostly True",
                           "Completely True", "Barely True", "Extremely False"]
            return output_list[prediction[0]]
