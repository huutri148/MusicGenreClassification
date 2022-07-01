import os.path
import json
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from src.data.load_data  import load_data
from src.Preprocess.ConvertToMFCC import deduct_mfcc
import numpy as np
from Data.MetaData.genre import mapping_genre

TEST_MFCC_FOLDER = r"../../Data/DataSet/TestMFCC/"
TEST_SONG = r"../../Data/DataSet/TestSong/"
MODEL_PATH = r"../../Models/Model_MFCC/Res50\\Model.h5"
PREDICTION_PATH = r"../../Data/DataSet/Prediction/"
PREDICTIONS_PATH = r"../../Data/DataSet/Predictions/"
SONG_PATH = r"../../Data/MetaData/streaming/"


def prepare_datasets(test_size, validation_size,fileName):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(fileName)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def prepare_test_data():
    if  os.path.exists(TEST_MFCC_FOLDER):
        # os.makedirs(TEST_MFCC_FOLDER)

        for dir in os.listdir(TEST_SONG):

            songDir = os.path.join(TEST_SONG, dir)
            label = dir
            data = {
                "label" :"",
                "mfcc": []
            }
            for song in os.listdir(songDir):
                songPath = os.path.join(songDir, song)
                songID = song[:-4]
                outputFileName = songID + ".json"
                outputPath = os.path.join(TEST_MFCC_FOLDER, outputFileName)


                mfcc = deduct_mfcc(songPath)
                data['mfcc'] += mfcc
                data['label'] = label

                with open(outputPath, 'w+') as f:
                    json.dump(data, f, indent=4)

                data = {
                    "label": "",
                    "mfcc": []
                }


def load_test_data():
    labels = []
    mfccs = []
    names = []



    for f in os.listdir(TEST_MFCC_FOLDER):
        filePath= os.path.join(TEST_MFCC_FOLDER, f)
        fileName = f[:-5]
        with open(filePath, "r") as fp:
            mfcc_json = json.load(fp)

        x = np.array(mfcc_json['mfcc'])
        x = x[..., np.newaxis]

        mfccs.append(x)
        labels.append(mfcc_json["label"])
        names.append(fileName)


        del mfcc_json
        fp.close()



    return labels, names, mfccs


def save_predictions(loaded_model):

    for file in os.listdir(SONG_PATH):
        fileName = file[:-5]
        filePath = os.path.join(SONG_PATH, file)

        mfcc = deduct_mfcc(filePath, "predict")
        prediction = loaded_model.predict(mfcc).tolist()
        outputPath = PREDICTIONS_PATH + "\\" +  fileName + ".json"

        with open(outputPath, 'w+') as f:
            json.dump(prediction, f, indent=4)


    print("DONE !!!!")


def save_prediction(loaded_model):
    labels, names, mfccs = load_test_data()

    # Display list of available test songs.
    print (np.unique(names))

    predictions_song = []
    predictions_label = []
    counts = []

    data = {
        "predictions_song":[],
        "predictions_name":[],
        "counts":[]
    }

    # Calculate the latent feature vectors for all the songs.
    for i in range(0, len(names)):
        if(names[i] not in predictions_label):
            predictions_label.append(names[i])
            test_image = mfccs[i]
            prediction = loaded_model.predict(test_image).tolist()
            predictions_song.append(prediction)
            counts.append(1)
        elif(names[i] in predictions_label):
            index = predictions_label.index(names[i])
            test_image = mfccs[i]
            prediction = loaded_model.predict(test_image).tolist()
            predictions_song[index] = predictions_song[index] + prediction
            counts[index] = counts[index] + 1

    data["predictions_song"] = predictions_song
    data["predictions_name"] = predictions_label
    data["counts"] = counts


    outputPath = PREDICTION_PATH + "\\prediction.json"

    with open(outputPath, 'w+') as f:
        json.dump(data, f, indent=4)


def load_predictions():
    data = {
        "predictions_song": [],
        "predictions_name": [],
        "counts":[]
    }

    fileName = PREDICTION_PATH + "\\prediction.json"

    with open(fileName, "r") as fp:
        mfcc_json = json.load(fp)

    data["predictions_song"] += mfcc_json["predictions_song"]
    data["predictions_name"] += mfcc_json["predictions_name"]
    data["counts"] += mfcc_json["counts"]

    del mfcc_json
    fp.close()

    predictions_song = np.array(data['predictions_song'])

    return predictions_song, data['predictions_name'],data['counts']


if __name__ == "__main__":
    # Load the trained model
    loaded_model = load_model(MODEL_PATH)
    save_prediction(loaded_model)
    #load_predictions()
    # prepare_test_data()