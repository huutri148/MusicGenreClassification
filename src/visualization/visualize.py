import os.path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

OUT_HISTORY = "../../Reports/History"

def plot_history(history,path,type,INDEX):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    filePath = os.path.join(path, type)

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    fileName = filePath + "\\Model_Trained_" + str(INDEX) +".jpg"

    plt.savefig(fileName)

def save_history(history,path,type,INDEX):
    filePath = os.path.join(path, type)

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    fileName = filePath + "\\training_history_" + str(INDEX) + ".csv"

    pd.DataFrame(history.history).to_csv(fileName)


def save_achirtectture(model, path, typeModel):
    filePath = os.path.join(path, typeModel)

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    fileName = filePath + "\\Model_Architecture.jpg"
    tf.keras.utils.plot_model(model, to_file= fileName, show_shapes=True, show_layer_names=True)



