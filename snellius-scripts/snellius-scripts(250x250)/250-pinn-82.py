import numpy as np
import tensorflow as tf
#from scipy.integrate import solve_ivp
import pandas as pd
import os
import deepxde as dde
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Concatenate, Input, Conv3D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.constraints import MinMaxNorm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as K
# Set default floating-point type to float64
tf.keras.backend.set_floatx('float32')

sorted_concatenated_csv = "final run concat 250x250.csv"
data = pd.read_csv(sorted_concatenated_csv)
#data.drop(columns=['zCOM'], inplace=True)
print(data.head())
data['time'] = (data['mcsteps'] / 10000).astype(int)
data = data[['time'] + [col for col in data.columns if col != 'time']]
data.drop(columns=['mcsteps'], inplace=True)
print(data)
cytokine_columns = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']
smallest_values = data[cytokine_columns].min()
largest_values = data[cytokine_columns].max()

print("Smallest values for each cytokine:")
print(smallest_values)
print("\nLargest values for each cytokine:")
print(largest_values)
#def replace_negative_with_zero(data):
 #   num_negative_values = (data < 0).sum().sum()
  #  data[data < 0] = 0

   # return num_negative_values

#cytokine_columns = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']

#for col in cytokine_columns:
 #   num_negatives = replace_negative_with_zero(data[col])
  #  print(f"Number of negative values replaced with 0 in '{col}': {num_negatives}")

# define cytokines
cytokines = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']

# Remove brackets and convert to float
for col in cytokines:
    data[col] = data[col].str.strip('[]').astype(float)

# get unique time values
unique_time = data['time'].unique()

arrays = {}

# iterate over unique time values
for time in unique_time:
    # filter data for current value of time
    data_time = data[data['time'] == time]
    
    # initialize 50x50x6 array for current value of time
    array = np.zeros((250, 250, len(cytokines)))
    
    # get X and Y coordinates
    x = data_time['xCOM'].astype(int)
    y = data_time['yCOM'].astype(int)
    
    # get cytokine concentrations
    concentrations = data_time[['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']].values
    
    # assign cytokine concentrations to corresponding position in array
    array[x, y, :] = concentrations
    
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

print(input_sequences.shape)
print(output_values.shape)

# Simulation parameters
nx = 100
true_size = 5
s_mcs = 60.0
h_mcs = 1 / 60.0
lineconv = true_size / nx
areaconv = true_size**2 / nx**2
volumeconv = (true_size**2 * 1) / (nx**2 * 1)

# Parameters for each cytokine
Dil8 = 2.09e-6 * s_mcs / areaconv
muil8 = 0.2 * h_mcs
keil8 = 234e-5 * volumeconv * h_mcs
kndnil8 = 1.46e-5 * volumeconv * h_mcs
thetanail8 = 3.024e-5 * volumeconv * h_mcs

Dil1 = 3e-7 * s_mcs / areaconv
muil1 = 0.6 * h_mcs
knail1 = 225e-5 * volumeconv * h_mcs

Dil6 = 8.49e-8 * s_mcs / areaconv
muil6 = 0.5 * h_mcs
km1il6 = 250e-5 * volumeconv * h_mcs

Dil10 = 1.45e-8 * s_mcs / areaconv
muil10 = 0.5 * h_mcs
km2il10 = 45e-5 * volumeconv * h_mcs

Dtnf = 4.07e-9 * s_mcs / areaconv
mutnf = 0.5 * 0.225 * h_mcs
knatnf = 250e-5 * volumeconv * h_mcs
km1tnf = 70e-5 * volumeconv * h_mcs

Dtgf = 2.6e-7 * s_mcs / areaconv
mutgf = 0.5 * (1 / 25) * h_mcs
km2tgf = 280e-5 * volumeconv * h_mcs

cellpresente = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

cellpresentndn = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
])

cellpresentna = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

cellpresentm1 = np.array([
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0
])

cellpresentm2 = np.array([
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])


# Optionally, print the arrays to verify
print("cellpresente:", cellpresente.shape)
print("cellpresentndn:", cellpresentndn.shape)
print("cellpresentna:", cellpresentna.shape)
print("cellpresentm1:", cellpresentm1.shape)
print("cellpresentm2:", cellpresentm2.shape)

# Convert the NumPy arrays to TensorFlow constants
cellpresente_tf = tf.constant(cellpresente, dtype=tf.float32)
cellpresentndn_tf = tf.constant(cellpresentndn, dtype=tf.float32)
cellpresentna_tf = tf.constant(cellpresentna, dtype=tf.float32)
cellpresentm1_tf = tf.constant(cellpresentm1, dtype=tf.float32)
cellpresentm2_tf = tf.constant(cellpresentm2, dtype=tf.float32)

# Define the parameters as TensorFlow constants
keil8_tf = tf.constant(keil8, dtype=tf.float32)
knail1_tf = tf.constant(knail1, dtype=tf.float32)
km1il6_tf = tf.constant(km1il6, dtype=tf.float32)
km2il10_tf = tf.constant(km2il10, dtype=tf.float32)
knatnf_tf = tf.constant(knatnf, dtype=tf.float32)
km2tgf_tf = tf.constant(km2tgf, dtype=tf.float32)
kndnil8_tf = tf.constant(kndnil8, dtype=tf.float32)
km1tnf_tf = tf.constant(km1tnf, dtype=tf.float32)
thetanail8_tf = tf.constant(thetanail8, dtype=tf.float32)

# diffusion degradation secretion endocytosis
D = np.array([Dil8, Dil1, Dil6, Dil10, Dtnf, Dtgf])
k = np.array([muil8, muil1, muil6, muil10, mutnf, mutgf])

# Stack the parameters for secretion and endocytosis
s1 = tf.stack([
    keil8_tf * cellpresente_tf,
    knail1_tf * cellpresentna_tf,
    km1il6_tf * cellpresentm1_tf,
    km2il10_tf * cellpresentm1_tf,
    knatnf_tf * cellpresentna_tf,
    km2tgf_tf * cellpresentm2_tf
], axis=0)

s2 = tf.stack([
    kndnil8_tf * cellpresentndn_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    km1tnf_tf * cellpresentm1_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32)
], axis=0)

e = tf.stack([
    thetanail8_tf * cellpresentna_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32)
], axis=0)

print("s1:", s1)
print("s2:", s2)
print("e:", e)


print("D:", D)
print("k:", k)
print("s1:", s1)
print("s2:", s2)
print("e:", e)

# Define the PDE
def pde(x, y):
    u = [y[:, i:i+1] for i in range(6)]
    laplacian_u = [dde.grad.hessian(u[i], x, i=1, j=1) + dde.grad.hessian(u[i], x, i=2, j=2) for i in range(6)]
    laplacian_u = tf.concat(laplacian_u, axis=1)
    degradation = k * y

    # Extract the time component from x
    time_idx = tf.cast(tf.floor(x[:, 2]), tf.int32)

    # Gather the appropriate values from s1, s2, and e based on time_idx
    secretion_1 = tf.gather(tf.transpose(s1), time_idx)
    secretion_2 = tf.gather(tf.transpose(s2), time_idx)
    endocytosis = tf.gather(tf.transpose(e), time_idx) * y

    return laplacian_u * D - degradation + secretion_1 + secretion_2 - endocytosis

# Define the geometry and time domain
geom = dde.geometry.Rectangle([0, 0], [250, 250])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define the initial condition
def initial_condition(x):
    return np.zeros((x.shape[0], 6))

# Define the Neumann boundary condition
def neumann_boundary(x, on_boundary):
    return on_boundary

def neumann_bc(x, y, nx, ny):
    grad = [dde.grad.jacobian(y, x, i=0, j=j) for j in range(3)]
    return sum([g * n for g, n in zip(grad, [nx, ny])])

# Initial and boundary conditions
ic = dde.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)
bc = dde.NeumannBC(geomtime, neumann_bc, neumann_boundary)

# Reshape input_sequences
input_sequences_reshaped = input_sequences.reshape(input_sequences.shape[0], -1)

# Flatten the output values
output_values_reshaped = output_values.reshape(output_values.shape[0], -1)

# Define train, validation, and test sizes
train_size = int(0.7 * input_sequences_reshaped.shape[0])
val_size = int(0.1 * input_sequences_reshaped.shape[0])
test_size = input_sequences_reshaped.shape[0] - train_size - val_size

# Split data
X_train = input_sequences_reshaped[:train_size]
X_val = input_sequences_reshaped[train_size:train_size + val_size]
X_test = input_sequences_reshaped[train_size + val_size:]
y_train = output_values_reshaped[:train_size]
y_val = output_values_reshaped[train_size:train_size + val_size]
y_test = output_values_reshaped[train_size + val_size:]

#train_size = int(0.7 * input_sequences_reshaped.shape[0])
#test_size = int(0.20 * input_sequences_reshaped.shape[0])
#val_size = input_sequences_reshaped.shape[0] - train_size - test_size

#X_train = input_sequences_reshaped[:train_size]
#X_test = input_sequences_reshaped[train_size:train_size + test_size]
#X_val = input_sequences_reshaped[train_size + test_size:]

##y_train = output_values_reshaped[:train_size]
#y_test = output_values_reshaped[train_size:train_size + test_size]
#y_val = output_values_reshaped[train_size + test_size:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Custom data set
data = dde.data.DataSet(
    X_train,
    y_train,
    X_val,
    y_val,
)

print("Custom data set defined")

# Construct the neural network with LAAF-10 relu activation
activation = "LAAF-10 relu"
net = dde.maps.FNN([input_sequences_reshaped.shape[1]] + [250] * 3 + [output_values_reshaped.shape[1]], activation, "Glorot uniform")

# Ensure the output layer uses ReLU to prevent negative predictions
net.apply_output_transform(lambda x, y: tf.nn.relu(y))

# Define the model
model = dde.Model(data, net)

# Define the PDE residual function
def pde_residual(x, y):
    return pde(x, y)

# Compile the model with combined losses: data loss (MSE) and PDE residual loss
model.compile("adam", lr=1e-3, metrics=["mean squared error"])

# Adding the physics-informed loss component directly into the training process
losshistory, train_state = model.train(iterations=10000, display_every=1000)

#metrics
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None)))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    abs_diff = np.abs(y_true - y_pred)
    threshold = 0.2 * np.abs(y_true)
    accurate_predictions = abs_diff <= threshold
    accuracy = np.mean(accurate_predictions)
    return accuracy


y_pred = model.predict(X_test)

# Reshape the predictions and true values
y_pred_reshaped = y_pred.reshape(y_test.shape[0], 250, 250, 6)
y_test_reshaped = y_test.reshape(y_test.shape[0], 250, 250, 6)

# Evaluate the model using various metrics
mape = mean_absolute_percentage_error(y_test_reshaped, y_pred_reshaped)
mse = mean_squared_error(y_test_reshaped, y_pred_reshaped)
mae = mean_absolute_error(y_test_reshaped, y_pred_reshaped)
msle = mean_squared_logarithmic_error(y_test_reshaped, y_pred_reshaped)
r2 = r_squared(y_test_reshaped, y_pred_reshaped)
acc = accuracy(y_test_reshaped, y_pred_reshaped)

print("Mean Absolute Percentage Error (MAPE) on the test set:", mape)
print("Mean Squared Error (MSE) on the test set:", mse)
print("Mean Absolute Error (MAE) on the test set:", mae)
print("Mean Squared Logarithmic Error (MSLE) on the test set:", msle)
print("R-squared (R²) on the test set:", r2)
print("Accuracy on the test set:", acc)

print("y_pred:", y_pred.shape)
print("y_pred_flat:", y_pred_reshaped.shape)

# Reshape predictions back to (19, 50, 50, 6)
y_pred_reshaped = y_pred.reshape(y_test.shape[0], 250, 250, 6)
y_test_reshaped = y_test.reshape(y_test.shape[0], 250, 250, 6)

print("Prediction completed")
print("y_pred reshaped:", y_pred_reshaped.shape)
print("y_test reshaped:", y_test_reshaped.shape)

y_pred = y_pred_reshaped
y_test = y_test_reshaped

y_pred_shape = y_pred.shape
y_test_shape = y_test.shape

# Flatten the y_pred and y_test tensors
y_pred_flattened = np.reshape(y_pred, (y_pred_shape[0], -1, y_pred_shape[-1]))
y_test_flattened = np.reshape(y_test, (y_test_shape[0], -1, y_test_shape[-1]))

# Create arrays for x and y coordinates
X = np.repeat(np.arange(y_test_shape[1]), y_test_shape[2])
Y = np.tile(np.arange(y_test_shape[2]), y_test_shape[1])

# Initialize a list to hold all data for the DataFrame
all_data = []

# Loop through each timestep and collect the data
for timestep in range(y_pred_shape[0]):
    y_pred_timestep = y_pred_flattened[timestep]
    for i in range(y_pred_timestep.shape[0]):
        data_point = {'timestep': timestep, 'X': X[i], 'Y': Y[i]}
        data_point.update({f'feature_{j+1}': y_pred_timestep[i, j] for j in range(y_pred_shape[-1])})
        all_data.append(data_point)

# Create the DataFrame
df_all_timesteps = pd.DataFrame(all_data)

# Save the DataFrame to a CSV file
df_all_timesteps.to_csv('PINN(250x250)(82-100hrs).csv', index=False)