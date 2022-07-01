import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import seaborn as sns
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix , ConfusionMatrixDisplay,accuracy_score

INPUT_MFCC = r"../../Data/DataSet/DatasetMFCC/Trained/"
INPUT_MODEL = r"../../Models/Model_MFCC/Res50\\Model.h5"
labels_genre = [
    "country",
    "dance-viet",
    "edm-viet",
    "electron-dance",
    "k-pop",
    "latin",
    "nhac-dan-ca-que-huong",
    "nhac-phim-us-uk",
    "nhac-phim-viet",
    "nhac-tru-tinh",
    "pop",
    "pop-ballad",
    "R-B-soul" ,
    "r-b-viet",
    "rap-viet",
    "rock",
    "trance-house-techno",
    "v-pop",
]
labels_genre_reverse = [
    "v-pop",
    "trance-house-techno",
    "rock",
    "rap-viet",
    "r-b-viet",
    "R-B-soul",
    "pop-ballad",
    "pop",
    "nhac-tru-tinh",
    "nhac-phim-viet",
    "nhac-phim-us-uk",
    "nhac-dan-ca-que-huong",
    "latin",
    "k-pop",
    "electron-dance",
    "edm-viet",
    "dance-viet",
    "country",
]

def load_data():
    data = {
        "mfcc": [],
        "labels": []
    }

    for filename in os.listdir(INPUT_MFCC):
        fullPath = os.path.join(INPUT_MFCC, filename)
        print(fullPath)
        with open(fullPath, "r") as fp:
            mfcc_json = json.load(fp)
        data["mfcc"] += mfcc_json["mfcc"]
        data["labels"] += mfcc_json["labels"]
        fp.close()


    # with open(data_path, "r") as fp:
    #     data = json.load(fp)
    print("Len mfcc: " + str(len(data["mfcc"])))
    print("Len labels: " + str(len(data["labels"])))

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y




def prepare_data_test():
    # load data
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

    X_test = X_test[..., np.newaxis]

    return X_test,y_test

def load_model_evaluate():
    # load model
    model = load_model(INPUT_MODEL)

    X_test, y_test = prepare_data_test()


    y_pred = model.predict(X_test, batch_size=32, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    #
    #
    # Print f1, precision, and recall scores
    print("Precision: {}".format(precision_score(y_test, y_pred_bool, average="macro")))

    print("F1 Score: ")
    print(f1_score(y_test, y_pred_bool, average="macro"))

    print("Recall Score: ")
    print(recall_score(y_test, y_pred_bool , average="macro"))
    #
    #
    #
    # ax = plt.subplot()
    # cm = confusion_matrix(y_test, y_pred_bool )
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(labels_genre, rotation = 45)
    # ax.yaxis.set_ticklabels(labels_genre_reverse, rotation = 0)
    #
    # disp.plot()
    # plt.show()
    # loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    #
    # print("Loss:")
    # print(loss)
    # print("Acc:")
    # print(accuracy)
    # print("f1_score: ")
    # print(f1_score)
    # print("precision: ")
    # print(precision)
    # print("recal: ")
    # print(recall)



if __name__ == "__main__":

    load_model_evaluate()