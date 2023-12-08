from dataLoader import dataLoader
import lieDetector
from uiLogic import uiLogic

def main():
    dl = dataLoader()
    ui = uiLogic()
    ui.updateUI()


if __name__ == '__main__':
    main()
