from transformers import BertTokenizer, BertForSequenceClassification


class lieDetector:

    def __init__(self):
        self.__var = 0
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def load(self):
        #code to load in model here
        placeholder = 0

    def predict(self, text): # add in a parameter for the text to be predicted
        #Does prediction based on data
        placeholder = 0
        return placeholder