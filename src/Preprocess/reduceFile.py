import os
import subprocess
from sys import path



FOL_WAV = r"../../Data/DataSet/DatasetWav30s/"
def reduceFile():
    for d in os.listdir(FOL_WAV):
        DRIVE_PATH = os.path.join(FOL_WAV, d)
        count = 0
        for file in os.listdir(DRIVE_PATH):
            if count > 500:
                filePath = os.path.join(DRIVE_PATH, file)
                os.remove(filePath)
            else:
                count+=1

if __name__ == "__main__":
    reduceFile()
