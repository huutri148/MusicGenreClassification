import os
import flask
from flask import Flask, request,jsonify
from flask_cors import CORS
import numpy as np
import json
import pyodbc

from dotenv import load_dotenv
load_dotenv()




MODEL_PATH = r"../../Models/Model_MFCC/"
RES50_PATH = MODEL_PATH + "Res50\\Model.h5"
RES18_PATH = MODEL_PATH + "Res18\\Model.h5"
RCNN_PATH = MODEL_PATH + "RCNN\\Model.h5"
PREDICTION_PATH = r"../../Data/DataSet/Predictions/"
PREDICTIONS_PATH = r"../../Data/DataSet/Prediction/"

recommend_genre = {
    0: ["IWZ9Z096"],
    1: ["IWZ9Z0CW"],
    2: ["IWZ97FCE"],
    3: ["IWZ9Z09A"],
    4: ["IWZ9Z08W", "IWZ9Z09O", "IWZ9Z09W"],
    5: ["IWZ97797"],
    6: ["IWZ9Z08D"],
    7: ["IWZ9Z0EC"],
    8: ["IWZ9Z0BA"],
    9: ["IWZ9Z08B"],
    10: ["IWZ9Z097"],
    11: ["IWZ9Z09W"],
    12: ["IWZ9Z09D"],
    13: ["IWZ9Z0C8"],
    14:["IWZ9Z089"],
    15:["IWZ9Z099"],
    16: ["IWZ9Z0C7"],
    17: ["IWZ97FCD"],
}


def load_predictions():
    data = {
        "predictions_song": [],
        "predictions_name": [],
        "counts":[]
    }

    fileName = PREDICTIONS_PATH + "\\prediction.json"

    with open(fileName, "r") as fp:
        mfcc_json = json.load(fp)

    data["predictions_song"] += mfcc_json["predictions_song"]
    data["predictions_name"] += mfcc_json["predictions_name"]
    data["counts"] += mfcc_json["counts"]

    del mfcc_json
    fp.close()

    predictions_song = np.array(data['predictions_song'])

    return predictions_song, data['predictions_name'],data['counts']


# Khởi tạo Flask Server Backend
app = Flask(__name__)
CORS(app)
# print("Number of GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))

# Load model và bộ recommended Song
cursor = {}
cursor2 = {}
#MODEL = load_model(RES50_PATH)
predictions_song, predictions_name, counts = load_predictions()


@app.route('/ping', methods=['GET'])
def testAPI():
    return {'msg':'pong'}




@app.route('/recommend/', methods=['GET'])
def getRecommend():
    name_file = request.args.get('name_file')
    songs = getRecommendedSong(name_file)
    return jsonify(songs)

@app.route('/userRecommend', methods=['GET'])
def getUserRecommend():
    userId = request.args.get('userId')
    songIDs = getTopUserSong(userId, cursor)
    songs = []
    for songID in songIDs:
        recommendSongs = getFilterRecommendedSong(songID,songs)
        songs = songs + recommendSongs
    return jsonify(songs)


@app.route('/genreRecommend', methods=['GET'])
def getGenreRecommend():
    userId = request.args.get('userId')
    songIDs = getTopUserSong(userId, cursor2)
    genres = []
    recommendGenre= []
    result = []
    for song in songIDs:
        gen = getFirstPrediction(song)
        if len(gen) != 0:
            genIndex = np.argmax(gen)
            if genIndex not in genres:
                genres.append(genIndex)
    for genre in genres:
        recommendGenre += recommend_genre[genre]

    for gen in recommendGenre:
        data = getGenreSong(gen)
        if data:
            result.append(data)

    return  jsonify(result)



@app.route("/models", methods = ['GET'])
def getListModel():
    models = []
    for dir in os.listdir(MODEL_PATH):
        models.append(dir)
    return {'models':models}


# @app.route("/currentModel", methods=['GET'])
# def currentModel():
#     if request.method == "GET":
#         return {"summary": MODEL.summary()}



def getSong(songID):
    result = cursor.execute("""
                          select *
                            from Song 
                           where SongID = ?
                          """,songID).fetchone()
    artResult = cursor.execute(
        """ SELECT * FROM Artist A Inner JOIN ArtistSong B ON A.ArtistID = B.ArtistID where B.SongID =  ?;""", songID) .fetchall()

    artists = []
    artistsName = ""
    index = 0
    for artist in artResult:
        obj = {
            "artistId": artist.ArtistID,
            "artistName": artist.ArtistName,
            "thumbnail": artist.Thumbnail,
            "isDeleted": artist.IsDeleted
        }
        artists.append(obj)
        if(index  !=  0):
            artistsName += ", " + artist.ArtistName
        else:
            artistsName += artist.ArtistName
        index +=1


    data = {
        "songId": result.SongID,
        "songName": result.SongName,
        "thumbnail": result.Thumbnail,
        "link": result.Link,
        "linkLyric": result.LinkLyric,
        "duration": result.Duration,
        "releaseDate": result.ReleaseDate,
        "likes": result.Likes,
        "downloads": result.Downloads,
        "listens": result.Listens,
        "isDeleted": result.IsDeleted,
        "isRecognizable": result.IsRecognizable,
        "artist":artists,
        "artistsName": artistsName

    }
    return data

def connectDB():
    global cursor
    global cursor2

    server = 'uit-music-player.database.windows.net'
    database = 'MUSICPLAYER'
    username = 'admin123'
    password = 'Newt1480'
    driver = '{ODBC Driver 17 for SQL Server}'

    # conn = pyodbc.connect('Driver={SQL Server};'
    #                       'Server=TEDDY\\MSSQLSERVER3;'
    #                       'Database=MUSICPLAYER;'
    #                       'TRANSLATE=1;'
    #                       'Trusted_Connection=yes;')
    # conn2 = pyodbc.connect('Driver={SQL Server};'
    #                       'Server=TEDDY\\MSSQLSERVER3;'
    #                       'Database=MUSICPLAYER;'
    #                       'TRANSLATE=1;'
    #                       'Trusted_Connection=yes;')
    conn = pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    conn2 = pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1434;DATABASE='+database+';UID='+username+';PWD='+ password)

    conn.setencoding(encoding='utf8')
    conn2.setencoding(encoding='utf8')

    cursor = conn.cursor()
    cursor2 = conn2.cursor()

    print("Connect to DB Succesfully!!!")

def getTopUserSong(userId, cursor):
    result = cursor.execute("""
                              select Top 3 *
                                from History
                               where UserID = ? and Count >= 15
                               Order by Count Desc;
                              """, userId).fetchall()
    songs = []
    for row in result:
        songs.append(row.SongID)

    return songs

def getPredictAnchor(fileName):

    prediction_anchor = np.full([10,18], 0)
    count = 0

    for i in range(0, len(predictions_name)):
        if(predictions_name[i] == fileName):
            prediction_anchor = prediction_anchor + predictions_song[i]
            count = counts[i]
            break

    # Count is used for averaging the latent feature vectors.
    prediction_anchor = prediction_anchor / count

    return prediction_anchor, count

def getRecommendedSong(fileName):
    if fileName in predictions_name:
        predict_anchor, count = getPredictAnchor(fileName)
        songs = recommend(predict_anchor, count, predictions_song, predictions_name, counts, fileName)
        songInfos = []
        for song in songs:
            songInfos.append(getSong(song))
        return songInfos
    else:
        print("Didn't have it in recommended song")
        file_path = PREDICTION_PATH + "\\" + fileName + ".json"
        if os.path.exists(file_path):
            predict_anchor = np.full([10,18], 0)
            count = 1

            with open(file_path, "r") as fp:
                prediction = json.load(fp)

            predict_anchor = predict_anchor + np.array(prediction)
            songInfos = []

            songs = recommend(predict_anchor, count, predictions_song, predictions_name, counts, fileName)
            for song in songs:
                songInfos.append(getSong(song))
            return songInfos
        else:
            return []

def getFilterRecommendedSong(fileName, songList):
    if fileName in predictions_name:
        predict_anchor, count = getPredictAnchor(fileName)
        songs = recommend(predict_anchor, count, predictions_song, predictions_name, counts, fileName)
        songInfos = []
        for song in songs:
            s = getSong(song)
            if s not in songList:
                songInfos.append(s)
        return songInfos
    else:
        print("Didn't have it in recommended song")
        file_path = PREDICTION_PATH + "\\" + fileName + ".json"
        if os.path.exists(file_path):
            predict_anchor = np.full([10,18], 0)
            count = 1

            with open(file_path, "r") as fp:
                prediction = json.load(fp)

            predict_anchor = predict_anchor + np.array(prediction)
            songInfos = []

            songs = recommend(predict_anchor, count, predictions_song, predictions_name, counts, fileName)
            for song in songs:
                s = getSong(song)
                if s not in songList:
                    songInfos.append(s)

            return songInfos
        else:
            return []

def getFirstPrediction(fileName):
    file_path = PREDICTION_PATH + "\\" + fileName + ".json"
    prec =[]
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            prediction = json.load(fp)
        prec = prediction[0]

    return prec

def getGenreSong(genreId):
    result = cursor2.execute("""select top 10 * from GenreSong A inner join Genre B on A.GenreID = B.GenreID
                                inner join Song on A.SongID = Song.SongID 
                                where A.GenreID = ?
                                order by Listens Desc;""",
                                genreId).fetchall()
    songs = []
    if len(result)> 0:
        for row in result:
            artistsName, artists = getArtist(row.SongID, cursor2)
            song = {
                "songId": row.SongID,
                "songName": row.SongName,
                "thumbnail": row.Thumbnail,
                "link": row.Link,
                "linkLyric": row.LinkLyric,
                "duration": row.Duration,
                "releaseDate": row.ReleaseDate,
                "likes": row.Likes,
                "downloads": row.Downloads,
                "listens": row.Listens,
                "isDeleted": row.IsDeleted,
                "isRecognizable": row.IsRecognizable,
                "artist": artists,
                "artistsName": artistsName
            }
            songs.append(song)

        data = {
            "genreId":genreId,
            "genreName": result[0].GenreName,
            "songs": songs
        }
        return data

def getArtist(songId, cursor):
    artResult = cursor.execute(
        """ SELECT * FROM Artist A Inner JOIN ArtistSong B ON A.ArtistID = B.ArtistID where B.SongID =  ?;""",
        songId).fetchall()

    artists = []
    artistsName = ""
    index = 0
    for artist in artResult:
        obj = {
            "artistId": artist.ArtistID,
            "artistName": artist.ArtistName,
            "thumbnail": artist.Thumbnail,
            "isDeleted": artist.IsDeleted
        }
        artists.append(obj)
        if (index != 0):
            artistsName += ", " + artist.ArtistName
        else:
            artistsName += artist.ArtistName
        index += 1

    return artistsName, artists

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





def runFlask():
    port = int(os.environ.get('PORT', 8089))
    app.run(host='0.0.0.0', port=port,ssl_context=('cert.pem', 'key.pem') )








# Start Backend
if __name__ == '__main__':
    connectDB()
    runFlask()
