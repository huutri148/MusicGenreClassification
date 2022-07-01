import os.path

from tensorflow import keras
from keras.models import load_model
import numpy as np
from src.data.make_dataset import load_test_data, load_predictions
from src.Preprocess.ConvertToMFCC import deduct_mfcc



SONG_PATH = r"../../Data/MetaData/streaming/"

MODEL_PATH = r"../../Models/Model_MFCC/Res50\\Model.h5"
#
# # Load the trained model
loaded_model = load_model(MODEL_PATH)
# #
# # print(loaded_model.summary())
#
#
# #labels, names, mfccs = load_test_data()
# predictions_song, predictions_name, counts = load_predictions()
#
# # Display list of available test songs.
# print (np.unique(predictions_name))
#
# # Enter a song name which will be an anchor song.
# recommend_wrt = "ZO8D7CO9"
#
# prediction_anchor = np.full([10,32], 0)
# count = 0
# distance_array = []
#
#
# for i in range(0, len(predictions_name)):
#     if(predictions_name[i] == recommend_wrt):
#         prediction_anchor = prediction_anchor + predictions_song[i]
#         count = counts[i]
#         break
#
#
#
# # Count is used for averaging the latent feature vectors.
# prediction_anchor = prediction_anchor / count
# for i in range(len(predictions_song)):
#     predictions_song[i] = predictions_song[i] / counts[i]
#     # Cosine Similarity - Computes a similarity score of all songs with respect
#     # to the anchor song.
#     distance_array.append(np.sum(prediction_anchor * predictions_song[i]) / (np.sqrt(np.sum(prediction_anchor**2)) * np.sqrt(np.sum(predictions_song[i]**2))))
#
# distance_array = np.array(distance_array)
# recommendations = 0
#
#
# print ("Recommendation is:")
#
# # Number of Recommendations is set to 2.
# while recommendations < 5:
#     index = np.argmax(distance_array)
#     value = distance_array[index]
#     print( "Song Name: " + predictions_name[index] + " with value = %f" % (value))
#     distance_array[index] =  - np.inf
#     recommendations = recommendations + 1



def recommend(prediction_anchor, count, predictions_song, predictions_name, counts, fileName):

    distance_array = []
    # Count is used for averaging the latent feature vectors.
    prediction_anchor = prediction_anchor / count
    for i in range(len(predictions_song)):
        predictions_song[i] = predictions_song[i] / counts[i]
        # Cosine Similarity - Computes a similarity score of all songs with respect
        # to the anchor song.
        distance_array.append(np.sum(prediction_anchor * predictions_song[i]) / (
                    np.sqrt(np.sum(prediction_anchor ** 2)) * np.sqrt(np.sum(predictions_song[i] ** 2))))

    distance_array = np.array(distance_array)
    recommendations = 0


    recommend_songs = []
    # Number of Recommendations is set to 2.
    while recommendations < 5:
        index = np.argmax(distance_array)
        value = distance_array[index]
        distance_array[index] = - np.inf
        if predictions_name[index] != fileName:
            recommend_songs.append(predictions_name[index])
            recommendations = recommendations + 1


    return recommend_songs





