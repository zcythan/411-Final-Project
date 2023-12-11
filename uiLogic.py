import tkinter as tk

class uiLogic:

    def __init__(self):
        self.__var = 0
        self.window = tk.Tk()
        self.prompt = tk.Label(self.window, text="Training Model, (May take up to 5 Minutes)...")
        self.prompt.pack()
        self.text_box = tk.Text(self.window, height=10, width=50)
        self.submit_button = tk.Button(self.window, text="Submit", command=self.submit_text)
        #self.lie_detector = lieDetector

    def mainWindow(self):
        self.prompt = tk.Label(self.window, text="Enter a statement from a political official to lie detect it")
        self.prompt.pack()
        self.text_box.pack()
        self.submit_button.pack()

    def submit_text(self):
        text = self.text_box.get("1.0", "end-1c")
        lie_detection_result = self.lie_detector.predict(text)
        self.result_label.config(text=lie_detection_result)

    def updateUI(self):
        self.window.mainloop()
        
    def update_after_data_load(self):
        self.prompt.config(text="Enter a statement from a political official to lie detect it")
        self.text_box.pack()
        self.submit_button.pack()
        self.result_label = tk.Label(self.window, text="Result will be shown here")
        self.result_label.pack()
        
    def schedule_update(self, func, *args):
        self.window.after(0, func, *args)