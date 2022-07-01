# CUT 30s to WAV
import os
import subprocess
from sys import path
import time
import datetime
import json
from Data.MetaData.genre import genre

listGenre = genre
currentGenre = ""

pathGenre= r"../../Data/MetaData/trainGenre/"
pathSongJSON = r"../../Data/MetaData/song/"

input = r"../../Data/MetaData/streaming/"
outFol = r"../../Data/DataSet/DatasetWav30s/"

bash_data_relative_path = r"../"
listSong ={}
totalSong = 0

total = 0
count = 0
lineBash = ""
bashAll = ""
index = 100




def writeBash(data):
    global index
    f = open("../../Data/BashCut/BashCUT30s/bash" +str(index)+".sh", 'a+')
    f.write(data)
    f.close()



def CUT(songId,file_path, output):
    #global total
    #dst_wav = output + ".wav"
    #sound = AudioSegment.from_mp3(file_path)
    global count
    global lineBash
    global bashAll
    global total
    global index
    try:
        duration = listSong[songId]
    except:
        duration = 0
    start = 15
    part = 1
    while start+30 < duration:
        begin_cut = str(datetime.timedelta(0, start))
        end_cut = str(datetime.timedelta(0, start+30))
        #print(begin_cut+"  "+end_cut)
        if not os.path.exists(output+"_"+str(part)+".wav"):
            # print(output+"_"+str(part)+".wav")
            lineBash = lineBash + ("ffmpeg -i "+ bash_data_relative_path +file_path+" -ss "+begin_cut +
                                   " -to "+end_cut+" "+ bash_data_relative_path +output+"_"+str(part)+".wav -loglevel error & ")
        start += 30
        part += 1
        total += 1
        count += 1
        if count >= 100:
            print(total)
            bashAll += "! "+lineBash + "\n"
            lineBash = ""
            count = 0
    if total >= 800:
        bashAll += "! "+lineBash + "\n"
        writeBash(bashAll)
        bashAll = ""
        total = 0
        lineBash = ""
        count = 0
        index += 1



def dumpSong(songPath):
    global totalSong
    with open(songPath, encoding="utf8") as fsong:
        line = fsong.readline()
        try:
            if totalSong <=3000:
                obj = json.loads(line)
                listSong[obj['encodeId']] = obj['duration']
                totalSong +=1
        except:
            pass

def dumpGenre():

    global currentGenre

    for f in os.listdir(pathGenre):

        fileName = os.path.join(pathGenre, f)
        currentGenre = f[:-4]

        outPath = os.path.join(outFol) + f[:-4]

        if not os.path.exists(outPath):
            os.makedirs(outPath)

        with open(fileName, encoding="utf8") as fGenre:
            lines = fGenre.readlines()

        for line in lines:
            print(line)
            songObj = json.loads(line)
            songPath = pathSongJSON + songObj['encodeId'] + ".txt"
            dumpSong(songPath)

            inFile = input + songObj['encodeId'] + ".mp3"
            outFile = outPath + "/" + songObj['encodeId']

            if os.path.exists(inFile):
                CUT(songObj['encodeId'], inFile, outFile)




if __name__ =="__main__":
    dumpGenre()
    #for fol in os.listdir(input):
    #    pathFol = os.path.join(input, fol)
    #    for filename in os.listdir(os.path.join(input, fol)):
    #        fullPath = os.path.join(pathFol, filename)
    #        # print(fullPath)
    #        CUT(filename[:-4],fullPath, os.path.join(outFol, fol, filename))

    #bashAll += "! "+lineBash + "\n"
    #writeBash(bashAll)