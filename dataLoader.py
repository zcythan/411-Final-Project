import deeplake
import threading
import queue

class dataLoader:
    def __init__(self):
        self.__trainSet = None
        self.__testSet = None
        self.__validSet = None
        self.packedTrain = None
        self.packedTest = None
        self.packedValid = None
        results_queue = queue.Queue()

        train_thread = threading.Thread(target=self.load, args=('train',))
        test_thread = threading.Thread(target=self.load, args=('test',))
        valid_thread = threading.Thread(target=self.load, args=('valid',))

        train_thread.start()
        test_thread.start()
        valid_thread.start()
        train_thread.join()
        test_thread.join()
        valid_thread.join()

        train_thread = threading.Thread(target=self.preload, args=(self.__trainSet, 'train'))
        test_thread = threading.Thread(target=self.preload, args=(self.__testSet, 'test'))
        valid_thread = threading.Thread(target=self.preload, args=(self.__validSet, 'valid'))

        train_thread.start()
        test_thread.start()
        valid_thread.start()
        train_thread.join()
        test_thread.join()
        valid_thread.join()
        print("Storing")
        print("Data stored")

    def load(self, type):
        if type == 'train':
            self.__trainSet = deeplake.load('hub://activeloop/liar-train')
        elif type == 'test':
            self.__testSet = deeplake.load('hub://activeloop/liar-test')
        elif type == 'valid':
            self.__validSet = deeplake.load('hub://activeloop/liar-val')

    def target(self, arg):
        result = self.preload(arg)
        self.results_queue.put(result)

    def preload(self, dataset, type):
        print("Packaging...")
        data_list = []
        tensorList = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation",
                      "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_onfire_counts", "context"]
        for sample in dataset:
            sample_dict = {}
            for tensor_name in tensorList:
                if tensor_name in sample:
                    tensor = sample[tensor_name]

                    # Converts tensor to NumPy array then standard array
                    np_tensor = tensor.numpy()
                    if np_tensor.ndim == 1 and np_tensor.size == 1:  # Single element
                        sample_dict[tensor_name] = np_tensor.item()
                    else:  # Array of elements or a single value as array
                        sample_dict[tensor_name] = np_tensor.tolist()

            data_list.append(sample_dict)
        print("Fully Packed")
        if type == 'train':
            self.packedTrain = data_list
        elif type == 'test':
            self.packedTest = data_list
        elif type == 'valid':
            self.packedValid = data_list



def parse(self):
    # parses data from DB
    placeholder = 0
