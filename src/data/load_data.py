import os
import json
import numpy as np
import re




def load_test_data(fileName):

    data = {
        "mfcc": [],
        "labels": []
    }

    with open(fileName, "r") as fp:
        mfcc_json = json.load(fp)
    data["mfcc"]+=mfcc_json["mfcc"]
    data["labels"]+=mfcc_json["labels"]
    del mfcc_json
    fp.close()

    print(len(data['mfcc']))



    print(len(data['mfcc']))


    print("Len mfcc: "+str(len(data["mfcc"])))
    print("Len labels: "+str(len(data["labels"])))


    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def load_data(fileName):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    data = {
        "mfcc": [],
        "labels": []
    }

    with open(fileName, "r") as fp:
        mfcc_json = json.load(fp)
    data["mfcc"]+=mfcc_json["mfcc"]
    data["labels"]+=mfcc_json["labels"]
    del mfcc_json
    fp.close()

    print(len(data['mfcc']))



    print(len(data['mfcc']))


    print("Len mfcc: "+str(len(data["mfcc"])))
    print("Len labels: "+str(len(data["labels"])))


    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y



if __name__ == "__main__":
    load_data(1)