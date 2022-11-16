import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings("ignore")

def dataload():
    df = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/MER_Audio/MER Audio/panda_dataset_taffc_annotations.csv")
    labels = list(df['Quadrant'].unique())
    path_list = []
    quad_list = []
    for i in range(len(df)):
        path = 'C:/Users/Saisamarth/Desktop/DA/MER_Audio/MER Audio/{}/{}.mp3'.format(df['Quadrant'][i],df['Song'][i])
        path_list.append(path)
        quad_list.append(df['Quadrant'][i])    
    files = {"Path" : path_list, "Class" : quad_list}
    data = pd.DataFrame(files)
    return data

def feature_extraction(df):
    features = []
    for i, r in df.iterrows():
        path = df["Path"][i]
        class_label = df["Class"][i]
        print(path)     
        audio, sample_rate = librosa.load(path, res_type = 'kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mean_mfcc = list(np.mean(mfcc,axis=1))
        features.append(mean_mfcc)     

    features_data = pd.DataFrame(features)
    features_data['Class'] = df['Class']     
    #return features_data
    top100 = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/Project/top100_features.csv")
    tdata = pd.concat([top100, features_data], axis = 1)
    return tdata

data = dataload()
tdata = feature_extraction(data)
tdata.to_csv('tdata.csv')
