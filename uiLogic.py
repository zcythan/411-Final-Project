import tkinter as tk

class uiLogic:

    def __init__(self):
        self.__var = 0
        self.window = tk.Tk()
        self.prompt = tk.Label(self.window, text="Enter a statement from a political official to lie detect it")
        self.prompt.pack()
        self.text_box = tk.Text(self.window, height=10, width=50)
        self.text_box.pack()
        self.submit_button = tk.Button(self.window, text="Submit", command=self.submit_text)
        self.submit_button.pack()

    def submit_text(self):
        text = self.text_box.get("1.0", "end-1c")
        print(text)  # replace this with your actual processing logic

    def updateUI(self):
        self.window.mainloop()