import os
import shutil
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
import json
import numpy as np
from src.models.CNN_architecture.res50_architecture import ResNet50
from src.visualization.visualize import plot_history, save_history,save_achirtectture

INDEX = 28
INPUT_MFCC = r"../../Data/DataSet/DatasetMFCC/Tested/"
TYPE = "Res50"
OUTPUT_MODEL = r"../../Models/Model_MFCC/" + TYPE +  r"\\Model.h5"
BEFORE_MODEL = r"../../Models/Model_MFCC/" + TYPE +  r"\\Model.h5"
OUTPUT_FOLDER = r"../../Models/Model_MFCC/"
OUTPUT_MFCC = r"../../Data/DataSet/DatasetMFCC/Passed/"

def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

def load_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    data = {
        "mfcc": [],
        "labels": []
    }

    for file in os.listdir(INPUT_MFCC):
          fullPath = os.path.join(INPUT_MFCC, file)

          with open(fullPath, "r") as fp:
                mfcc_json = json.load(fp)
          data["mfcc"] += mfcc_json["mfcc"]
          data["labels"] += mfcc_json["labels"]
          print("File: {}, MFCC: {}, Labels: {}".format(file,len(mfcc_json["mfcc"]), len(mfcc_json["labels"])))
          del mfcc_json
          fp.close()



    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_datasets(test_size, validation_size):
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
    X, y = load_data()

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train():
    global INDEX
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2)

    # # create network
    #input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if not os.path.exists(BEFORE_MODEL):
        print("New Model!")
        modelRes = ResNet50()
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        modelRes.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        #save_achirtectture(modelRes, OUTPUT_FOLDER, TYPE)

    else:
        print(BEFORE_MODEL)
        modelRes = load_model(BEFORE_MODEL)

    history = modelRes.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    #plot accuracy/error for training and validation
    plot_history(history,OUTPUT_FOLDER, TYPE,INDEX)

    save_history(history,OUTPUT_FOLDER, TYPE, INDEX)

    modelRes.save(OUTPUT_MODEL)

    # evaluate model on test set
    test_loss, test_acc = modelRes.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy1:', test_acc)



def trained_Model(fileName, model):
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2, fileName)



    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    # plot accuracy/error for training and validation
    plot_history(history, OUTPUT_FOLDER, TYPE, INDEX)

    save_history(history, OUTPUT_FOLDER, TYPE, INDEX)

    model.save(OUTPUT_MODEL)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy1:', test_acc)

    shutil.move(fileName, OUTPUT_MFCC)

    return 1;

def Test(fileName):
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2, fileName)

    modelRes = load_model(BEFORE_MODEL)

    # pick a sample to predict from the test set
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(modelRes, X_to_predict, y_to_predict)




if __name__ == "__main__":

    train()


    print("DONE!!!!")

    # testFile = r"../../Data/DataSet/DatasetMFCC/Passed\\MFCC_1_0.json"
    # Test(testFile)

