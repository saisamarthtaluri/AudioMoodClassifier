import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def pred_feature_index(path):
    tdata = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/Project/tdata.csv")
    songid = path[-16:-4]
    findex = tdata.index[tdata['IDSong']==songid].tolist()
    return findex

#findex = pred_feature_index()

def predictor(findex):
    tdata = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/Project/tdata.csv")
    fdata = tdata.drop(columns = 'IDSong')
    #findex=pred_feature_index(path)
    X = fdata.iloc[: ,:-1].values
    scaler = StandardScaler()
    feature_array = scaler.fit_transform(X)
    model = load_model('model.h5')
    res = model.predict(feature_array)[findex]
    
    class_num = np.argmax(res)
    if class_num == 0:
        class_label = 'Happy'
    elif class_num == 1:
        class_label = 'Angry'
    elif class_num == 2:
        class_label = 'Sad'
    else:
        class_label = 'Relaxed'
    
    return class_label


    