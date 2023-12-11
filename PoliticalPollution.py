from dataLoader import dataLoader
from lieDetector import lieDetector
from uiLogic import uiLogic
import threading
from threading import Lock
lock = Lock()
#Driver file
'''
def dataLoad(shared_data, lock):
    try:
        AI = lieDetector()  # Assuming this operation is thread-safe and error-free
        with lock:
            shared_data["AI"] = AI
            print("AI object stored in shared_data")  # For debugging
    except Exception as e:
        print(f"Error in dataLoad: {e}")

def check_data_and_update_ui(shared_data, ui):
    while "AI" not in shared_data:
        pass  # Replace with a more efficient mechanism
    ui.schedule_update(ui.update_after_data_load)  # Assuming this is correctly implemented

def main():
    shared_data = {}
    ui = uiLogic()

    data_thread = threading.Thread(target=dataLoad, args=(shared_data, lock))
    data_thread.start()

    ui_update_thread = threading.Thread(target=check_data_and_update_ui, args=(shared_data, ui))
    ui_update_thread.start()

    ui.updateUI()

    data_thread.join()
    ui_update_thread.join()

    if "AI" in shared_data:
        detector = shared_data["AI"]
        detector.train()  # Ensure this is implemented correctly
        prediction = detector.predict("Some statement to predict")
        print("Prediction:", prediction)
    else:
        print("AI object is not available in shared_data.")

if __name__ == '__main__':
    main()


'''


# Create a function that will be passed into a thread to handle the dataloading
def dataLoad(shared_data, lock):
    data = dataLoader()

    with lock: 
        shared_data["data"] = data

        
def check_data_and_update_ui(shared_data, ui):
    while "data" not in shared_data:
        pass  # Busy waiting; consider using a more efficient wait-notify mechanism

    ui.schedule_update(ui.update_after_data_load)

def main():
    shared_data = {}
    #detect = lieDetector()
    ui = uiLogic()
    aimod = lieDetector()
    #data_thread = threading.Thread(target=dataLoad, args=(shared_data, lock))
    #data_thread.start()
    
    #ui_update_thread = threading.Thread(target=check_data_and_update_ui, args=(shared_data, ui))
    #ui_update_thread.start()
    
    #ui.updateUI()

    #data_thread.join()
    #data = shared_data["data"]
    #ui_update_thread.join()

    #AI.train()  # Train the model
    #prediction = AI.predict("Some statement to predict")  # Make a prediction
    

if __name__ == '__main__':
    main()
