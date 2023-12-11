import random
import time


def dummyData(text):
    data_options = ["False", "Half-True", "Mostly-True",
                    "Completely True", "Barely True", "Super False"]
    time.sleep(5)
    choice = random.randint(0, 5)
    return data_options[choice]
