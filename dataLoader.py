import deeplake
from torchvision import transforms
from torch.utils.data import DataLoader


class dataLoader:
    def __init__(self):
        self.__trainSet = deeplake.load('hub://activeloop/liar-train', reset=True)
        self.preload()

    def preload(self):
        data_list = []
        tensorList = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation",
                      "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_onfire_counts", "context"]
        for sample in self.__trainSet:
            sample_dict = {}
            for tensor_name in tensorList:
                if tensor_name in sample:
                    tensor = sample[tensor_name]

                    # Convert tensor to NumPy array, then to Python native type
                    np_tensor = tensor.numpy()
                    if np_tensor.ndim == 1 and np_tensor.size == 1:  # Single element
                        sample_dict[tensor_name] = np_tensor.item()
                    else:  # Array of elements or a single value as array
                        sample_dict[tensor_name] = np_tensor.tolist()

            data_list.append(sample_dict)
        print(data_list)
        return data_list



def parse(self):
    # parses data from DB
    placeholder = 0
