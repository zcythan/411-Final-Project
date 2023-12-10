from dataLoader import dataLoader
import lieDetector
from uiLogic import uiLogic
import threading
from threading import Lock

#Driver file

lock = Lock()

# Create a function that will be passed into a thread to handle the dataloading
def dataLoad(shared_data, lock):
    dl = dataLoader()
    with lock: 
        shared_data["data"] = dl 
        
def check_data_and_update_ui(shared_data, ui):
    while "data" not in shared_data:
        pass  # Busy waiting; consider using a more efficient wait-notify mechanism

    ui.schedule_update(ui.update_after_data_load)

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
    


if __name__ == '__main__':
    main()
