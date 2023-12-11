from dataLoader import dataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load English stopwords
stop_words = set(stopwords.words('english'))
class lieDetector:
    def __init__(self):
        data_loader = dataLoader()
        print('Data loaded')
        self.vectorizer = TfidfVectorizer()
        # Extract features and labels from training, test, and validation sets
        value_train, labels_train = self.extract(data_loader.packedTrain, fit = True)
        value_test, labels_test = self.extract(data_loader.packedTest, fit = False)
        #value_valid, labels_valid = self.extract(data_loader.packedValid)

        print('Extraction complete')
        print("Training Model")

        self.model = RandomForestClassifier()
        self.model.fit(value_train, labels_train)

        # Predict and evaluate
        #predictions = self.model.predict(value_test)
        # ... evaluate predictions ...
        # Calculate accuracy on the test set
        accuracy = self.model.score(value_test, labels_test)
        print("Accuracy on test set:", accuracy)


    def predict(self, inputs):
        processed_input = ' '.join(
            word.lower() for word in inputs.split() if word.lower() not in stop_words)

        # Transform the processed_input using the vectorizer

        vectorized_statement = self.vectorizer.transform([processed_input])

        # Convert the result of transform to a dense array
        vectorized_statement_array = vectorized_statement.toarray()

        # Use the dense array for prediction
        print("Predicting for: " + inputs)
        prediction = self.model.predict(vectorized_statement_array)
        print(prediction[0])

    def extract(self, mode, fit = False):
        text_data = []
        numerical_data = []
        labels = []

        for dictionary in mode:
            '''
            #Normal classification
            if 'label' in dictionary:
                if fit == False:
                    if dictionary['label'] == 0:
                        dictionary['label'] = 3
                    elif dictionary['label'] == 1:
                        dictionary['label'] = 0
                    elif dictionary['label'] == 2:
                        dictionary['label'] = 1
                    elif dictionary['label'] == 3:
                        dictionary['label'] = 5
                    elif dictionary['label'] == 5:
                        dictionary['label'] = 2
            '''
            #Binary classification:
            if 'label' in dictionary:
                if fit == False:
                    if dictionary['label'] == 0:
                        dictionary['label'] = 3
                    elif dictionary['label'] == 1:
                        dictionary['label'] = 0
                    elif dictionary['label'] == 2:
                        dictionary['label'] = 1
                    elif dictionary['label'] == 3:
                        dictionary['label'] = 5
                    elif dictionary['label'] == 5:
                        dictionary['label'] = 2

                if dictionary['label'] in [0, 1, 4, 5]:
                    dictionary['label'] = 0
                elif dictionary['label'] in [2, 3]:
                    dictionary['label'] = 1


                labels.append(dictionary.pop('label'))

            # Extract text and numerical data separately
                # Extract and preprocess text data
                text_values = ' '.join(str(dictionary[key]) for key in
                                       ["statement", "subject", "speaker", "job_title", "state_info",
                                        "party_affiliation", "context"] if key in dictionary)

                # Lowercase and remove stopwords
                processed_text = ' '.join(
                    word.lower() for word in text_values.split() if word.lower() not in stop_words)

                text_data.append(processed_text)

            num_keys = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
                        "pants_onfire_counts"]
            num_values = {key: dictionary.get(key, 0) for key in num_keys}
            numerical_data.append(num_values)

        # Vectorize text data
        #vectorizer = TfidfVectorizer()
        #tfidf_matrix = vectorizer.fit_transform(text_data)
        if fit:
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
        else:
            tfidf_matrix = self.vectorizer.transform(text_data)

        # Scale numerical data
        numerical_data_df = pd.DataFrame(numerical_data).fillna(0)

        # Ensure all data is numeric
        numerical_data_df = numerical_data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        # Combine vectorized text data with scaled numerical data
        combined_features = hstack([tfidf_matrix, numerical_data_df])

        return tfidf_matrix.toarray(), np.array(labels)



