import deeplake
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
class dataLoader:
    def __init__(self):
        self.__trainSet = deeplake.load('hub://activeloop/liar-train')
        self.preload()

    def preload(self):
        print("Preloading data...")
        for i, sample in enumerate(self.__trainSet):
            # Access data in each tensor
            id = sample['id'].numpy()
            label = sample['label'].numpy()
            statement = sample['statement'].numpy()

            # Print or process the data
            print(f"Sample {i}: ID - {id}, Label - {label}, Statement - {statement}")

            if i >= 5:  # Display first 5 samples
                break
        print("Preloading complete.")

    def parse(self):
        #parses data from DB
        placeholder = 0