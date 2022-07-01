import json
import os
import math
from re import T
import librosa
import threading
import numpy as np
from Data.MetaData.genre import obj_genre

labels_genre = {
    "country": 0,
    "dance-viet": 1,
    "edm-viet": 2,
    "electron-dance": 3,
    "k-pop": 4,
    "latin": 5,
    "nhac-dan-ca-que-huong": 6,
    "nhac-phim-us-uk": 7,
    "nhac-phim-viet": 8,
    "nhac-tru-tinh": 9,
    "pop": 10,
    "pop-ballad": 11,
    "R-B-soul": 12,
    "r-b-viet": 13,
    "rap-viet": 14,
    "rock": 15,
    "trance-house-techno": 16,
    "v-pop": 17,

}

GENRE = "IWZ9Z0BA"
#FOL_WAV = r"F:/Data/wav30s/"
FOL_WAV = r"../../Data/DataSet/DatasetWav30s/"
FOL_OUT_MFCC = r"../../Data/DataSet/DatasetMFCC1/"
#wavDir = r"/media/huutri148/Teddy/Data/wav30s"
wavDir = r"../../Data/DataSet/DatasetWav30s/"

AUDIO_PER_FILE = 200



SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(LAST_HIT, MIN_INDEX_FILE, MAX_INDEX_FILE, num_mfcc=32, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "labels": [],
        "mfcc": []
    }
    print(str(MIN_INDEX_FILE))
    total_audio = 0
    index_file = MIN_INDEX_FILE
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for f in os.listdir(FOL_WAV):
        # load audio file
        if total_audio >= MIN_INDEX_FILE * AUDIO_PER_FILE and total_audio < MAX_INDEX_FILE * AUDIO_PER_FILE:
            try:
                file_path = os.path.join(FOL_WAV, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        # print(num_mfcc_vectors_per_segment)
                        data["mfcc"].append(mfcc.tolist())
                        #data["labels"].append(STATIC_LABEL)
                        # print("{}, segment:{}".format(file_path, d+1))
            except:
                print("Error")

            if total_audio % AUDIO_PER_FILE == AUDIO_PER_FILE - 1:
                print("Dump file :" + str(index_file))
                with open(FOL_OUT_MFCC + "/" + GENRE + str(index_file) + ".json", "w+") as fp:
                    json.dump(data, fp, indent=4)
                index_file += 1
                data = {
                    "labels": [],
                    "mfcc": []
                }
        total_audio += 1
    if LAST_HIT == True:
        print("Dump file :" + str(index_file))
        with open(FOL_OUT_MFCC + "/" + GENRE + str(index_file) + ".json", "w+") as fp:
            json.dump(data, fp, indent=4)

def convert_MFCC(listDir,threadID, num_mfcc=32, n_fft=2048, hop_length=512, num_segments=10):

    print("ID of Thread: {}, {}".format(threadID, len(listDir)))
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "labels": [],
        "mfcc": []
    }
    total_audio = 0
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    index_file = 0

    for dir in listDir:
        #genre = str(dir)[-9:-1]
        wdir = os.path.join(FOL_WAV, dir)

        for f in os.listdir(wdir):
            # load audio file
            try:
                file_path = os.path.join(wdir, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        # print(num_mfcc_vectors_per_segment)
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(dir)
                        # print("{}, segment:{}".format(file_path, d+1))
            except :
                print("Error")

            if total_audio % AUDIO_PER_FILE == AUDIO_PER_FILE - 1:
                print("Dump file :" + str(index_file))
                with open(FOL_OUT_MFCC + "MFCC_" + str(threadID) + "_" + str(index_file) + ".json", "w+") as fp:
                    json.dump(data, fp, indent=4)
                index_file += 1
                data = {
                    "labels": [],
                    "mfcc": []
                }
            total_audio += 1


    print("Dump file :" + str(index_file))
    with open(FOL_OUT_MFCC + "MFCC_" + str(threadID) + "_" + str(index_file) + ".json", "w+") as fp:
        json.dump(data, fp, indent=4)


def deduct_mfcc(filePath, mode="save", num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    list_mfcc = []
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    # load audio file
    signal, sample_rate = librosa.load(str(filePath), sr=SAMPLE_RATE)
    duration = librosa.get_duration(signal, sample_rate)
    part = int((duration - 15) / 30)
    print("Duration: " + str(duration))

    if mode == "save":
        selectedPart =  math.ceil(5/2) - 1;
        mfcc_part = []
        for d in range(num_segments):
            start = samples_per_segment * (d + selectedPart * 10)
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(
                signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_part.append(mfcc.tolist())
        list_mfcc += mfcc_part
    else:
        # process all segments of audio file
        selectedPart = math.ceil(5 / 2) - 1;
        mfcc_part = []
        for d in range(num_segments):
            start = samples_per_segment * (d + selectedPart * 10)
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(
                signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_part.append(mfcc.tolist())
            # print(np.array(data["mfcc"]))
        x = np.array(mfcc_part)
        x = x[..., np.newaxis]
        list_mfcc.append(x)
    return list_mfcc



def count_file():
    listDir = []
    count = 0
    f = open("../../Data/MetaData\\total2.txt", 'a+', encoding='utf-8')
    for d in os.listdir(FOL_WAV):
        listDir.append(FOL_WAV + d + "/")
        DRIVE_PATH = os.path.join(FOL_WAV, d)
        numberFiles = next(os.walk(DRIVE_PATH))[2]
        count += len(numberFiles)

        f.write("{} - numberOfAudios: {} files\n".format( d, len(numberFiles)))

    f.write("Total Audio Files: {} files".format(count))


if __name__ == "__main__":
    #count_file()


    listDir = []
    for d in os.listdir(FOL_WAV):
        listDir.append(d)

    list1, list2, list3=  np.array_split(listDir, 3)

    threads =3
    t1 = threading.Thread(target=convert_MFCC, args=(list1,1))
    t2 = threading.Thread(target=convert_MFCC, args=(list2,2))
    t3 = threading.Thread(target=convert_MFCC, args=(list3,3))

    #t1 = threading.Thread(target=save_mfcc, args=(False, 0, 2))
    #t2 = threading.Thread(target=save_mfcc, args=(False, 2, 4))
    #t3 = threading.Thread(target=save_mfcc, args=(False, 4, 6))
    #t4 = threading.Thread(target=save_mfcc, args=(False, 6, 8))

    jobs = []

    jobs.append(t1)
    jobs.append(t2)
    jobs.append(t3)
    # Start the threads (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

    # Ensure all of the threads have finished
    for j in jobs:
        j.join()
    # save_mfcc(MIN_INDEX_FILE=6,MAX_INDEX_FILE=8,num_segments=10,)