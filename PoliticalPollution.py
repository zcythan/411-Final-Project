from dataLoader import dataLoader
import lieDetector
from uiLogic import uiLogic

#Driver file

def main():
    ui = uiLogic()
    #ui.updateUI()
    dl = dataLoader()
    ui.mainWindow()


if __name__ == '__main__':
    main()
