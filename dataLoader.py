import deeplake
import threading
from nltk.corpus import stopwords

class dataLoader:
    def __init__(self, type):
        self.stop_words = set(stopwords.words('english'))
        self.__trainSet = None
        self.__testSet = None
        self.packedTrain = None
        self.packedTest = None
        self.testLabels = None
        self.trainLabels = None
        self.classify = type

        train_thread = threading.Thread(target=self.load, args=('train',))
        test_thread = threading.Thread(target=self.load, args=('test',))

        train_thread.start()
        test_thread.start()
        train_thread.join()
        test_thread.join()

        train_thread = threading.Thread(
            target=self.preload, args=(self.__trainSet, 'train'))
        test_thread = threading.Thread(
            target=self.preload, args=(self.__testSet, 'test'))

        train_thread.start()
        test_thread.start()
        train_thread.join()
        test_thread.join()

        print("Storing")
        print("Data stored")

    def load(self, type):
        if type == 'train':
            self.__trainSet = deeplake.load('hub://activeloop/liar-train')
        elif type == 'test':
            self.__testSet = deeplake.load('hub://activeloop/liar-test')

    def preload(self, dataset, type):
        print("Packaging...")

        dataList = []
        dataLabels = []

        #Keys being used for training
        keys = ["statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "context"]

        for sample in dataset:
            strings = []

            # Adjust the label scheme if it's dealing with the test data, convert all labels to binary classification if needed
            #For whatever the reason, the numerical representations for true and false do not correlate between
            #the testing set and the training set. This was written to remap based on the observed difference between the two,
            #and it seems to work quite nicely.
            if 'label' in sample:
                label = sample['label'].numpy().item()
                if type == 'test':
                    if label == 0:
                        label = 3
                    elif label == 1:
                        label = 0
                    elif label == 2:
                        label = 1
                    elif label == 3:
                        label = 5
                    elif label == 5:
                        label = 2
                # Apply binary classification conversion
                if self.classify:
                    if label in [0, 1, 4, 5]:
                        dataLabels.append(0)
                    elif label in [2, 3]:
                        dataLabels.append(1)
                else:
                    dataLabels.append(label)

            # Loop through the keys in the dataset to pick out the needed features.
            for key in keys:
                if key in sample:
                    tensor = sample[key]

                    # Convert tensor to NumPy array
                    nptensor = tensor.numpy()
                    if nptensor.ndim == 1 and nptensor.size == 1:  # Single element
                        text = str(nptensor.item())
                    else:  # Array of elements or a single value as array
                        text = str(nptensor.tolist())

                    # move to lowercase and remove stopword, this boosted accuracy by about 2%
                    processedText = ' '.join(word.lower() for word in text.split() if word.lower() not in self.stop_words)
                    strings.append(processedText)

                # build the strings
            concatString = ' '.join(strings)
            dataList.append(concatString)
        print("Fully Packed")
        if type == 'train':
            self.packedTrain = dataList
            self.trainLabels = dataLabels
        elif type == 'test':
            self.packedTest = dataList
            self.testLabels = dataLabels


