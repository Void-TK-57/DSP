import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import librosa

from backend import ANN_Model as Model

import sys

import os

# function to get mfccs
def extract_features(file_name, n_mfcc = 40):
    print("Exctracting for: " + str(file_name))
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed


# method to plot history
def plot_history(history):
    # create subplots
    fig, axes = plt.subplots(len(history.keys()), 1)
    index = 0
    # for each key
    for key in history.keys():
        axes[index].plot(history[key], color="green")
        axes[index].set_xlabel("Epochs")
        axes[index].set_ylabel( str(key) )
        index += 1
    plt.show()

def main_read(path):

    dataframe = pd.read_csv(path, index_col=0)

    # create model
    model = Model(40, 10, [50, 40], ["sigmoid", "relu", "softmax"], 'adam', 0.001, ['accuracy'], "categorical_crossentropy")
    
    #features index
    features_index = list( [str(i) for i in range(40) ] ) + ["file"]
    # x values
    x_values = dataframe[features_index].values

    y_values = to_categorical( dataframe["Digito"].values, 10)

    # split into x values and y values
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, train_size = 0.7)

    history = model.train(x_train[:, :-1], y_train, epochs = 200).history

    plot_history(history)

    results = model.evaluate(x_test[:, :-1], y_test)

    y_predict = model.predict_classes(x_test[:, :-1])

    files = x_test[:, -1]
    print(len(files))

    print(len(y_predict))

    result = pd.DataFrame( np.stack( [files, y_predict], axis = 1), columns = ["File", "Predicao"])

    result.to_csv("predicao.csv")
    
    print(result)

def main_save(path = "../../data/8bit-C4.wav"):

    # get data
    n_mfcc = 40
    data = os.listdir('../../data/recordings')
    data = list([ i.replace('.wav', '').split("_") + [i,] for i in data])

    # create dataframe from data
    dataframe = pd.DataFrame(data, columns = ["Digit", "Voice Actor", "Replic", "Recording"])
    # load mfccs of each recording
    mffcs = dataframe["Recording"].apply(lambda path: extract_features("../../data/recordings/" + path, n_mfcc) )
    mffcs =  np.stack( list(mffcs.values), axis = 0)

    total_values = np.concatenate([dataframe.values, mffcs], axis=1)

    dataframe = pd.DataFrame(total_values)

    print(dataframe)

    dataframe.to_csv("test.csv")

    return

    


    
    
    
    
    
    
    
    


if __name__ == "__main__":
    # get args
    args = sys.argv
    # get path

    file = "8bit-C4.wav" if len(args) == 1 else args[1]
    # main("../../data/" + file)

    main_read(file)