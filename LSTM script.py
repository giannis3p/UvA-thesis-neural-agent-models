import tensorflow as tf
import pandas as pd
import numpy as np
import os
import csv
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.constraints import MinMaxNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

output_folder = 'C:/Users/Ioannis/Documents/UvA thesis/UvA-thesis/data/all_data'

def load_and_drop_zCOM(folder_path):
    df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.csv')])
    df.drop(columns=['zCOM'], inplace=True)
    return df

dfs = []
for i in range(1, 9):
    sim_folder_path = os.path.join(output_folder, f'S{i}')
    df = load_and_drop_zCOM(sim_folder_path)
    dfs.append(df)

df1, df2, df3, df4, df5, df6, df7, df8 = dfs

df8['time'] = (df8['mcsteps'] / 10000).astype(int)
df8 = df8[['time'] + [col for col in df8.columns if col != 'time']]
df8.drop(columns=['mcsteps'], inplace=True)
print(df8)

# define cytokines
cytokines = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']

# get unique time values
unique_time = df8['time'].unique()

arrays = {}

# iterate over unique time values
for time in unique_time:
    # filter data for current value of time
    df_time = df8[df8['time'] == time]
    
    # initialize 500x500 array for current value of time
    array = np.zeros((500, 500, len(cytokines)))
    
    # iterate over rows in filtered df
    for index, row in df_time.iterrows():
        # get X and Y coordinates
        x = int(row['xCOM'])
        y = int(row['yCOM'])
        
        # get cytokine concentrations
        concentrations = row[['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']].values
        
        # assign cytokine concentrations to corresponding position in array
        array[x, y] = concentrations
    
    # store array for current value of time
    arrays[time] = array


sequence_length = 10
input_sequences = []
output_values = []

# convert dictionary values to a list of arrays
arrays_list = [arrays[key] for key in sorted(arrays.keys())]

# convert 'arrays' list to numpy array
arrays_np = np.array(arrays_list)

for i in range(len(arrays_np) - sequence_length):
    input_seq = arrays_np[i:i+sequence_length]  # input sequence of arrays
    output_val = arrays_np[i+sequence_length]   # array at next time step
    
    input_sequences.append(input_seq)
    output_values.append(output_val)

# convert lists to numpy arrays
input_sequences = np.array(input_sequences)
output_values = np.array(output_values)


model = Sequential()
model.add(LSTM(units=64, input_shape=(10, 500 * 500 * 6)))  # 10 for a sequence length of 10 as defined above
model.add(Dense(units=100, activation='relu'))  # 100 neurons, first hidden layer, relu
model.add(Dense(units=100, activation='relu'))  # 100 neurons, second hidden layer, relu
model.add(Dense(units=500 * 500 * 6, activation='linear'))  # output layer, linear activation
model.add(Reshape((500, 500, 6)))
model.compile(optimizer='adam', loss='mse')  # compile with adam, mse
print(model.summary())

input_sequences_reshaped = input_sequences.reshape(input_sequences.shape[0], 10, -1)

# train
history = model.fit(input_sequences_reshaped, output_values, epochs=10, batch_size=32, validation_split=0.2)
print("Training Loss:", history.history['loss'])

# evaluate
loss = model.evaluate(input_sequences, output_values)
print("Test Loss:", loss)
