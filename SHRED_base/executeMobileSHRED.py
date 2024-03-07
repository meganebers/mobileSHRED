
import numpy as np
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
from processdata import TimeSeriesDataset
import models
from models import fit
import torch

# Define constants
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 2
L1_REGULARIZATION = 350
L2_REGULARIZATION = 400
DROPOUT_RATE = 0.1
BATCH_SIZE = 512
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
PATIENCE = 5

def mse_error(Ypred, Ytest):
    err = np.linalg.norm(Ypred - Ytest) / np.linalg.norm(Ytest) 
    return err


def partition_data(load_X, n, lags):
    #np.random.seed(0)
    train_indices = np.random.choice(n - lags, size=int(n*0.60), replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    return train_indices, valid_indices, test_indices


def transform_data(load_X, train_indices):
    # sklearn's MinMaxScaler is used to preprocess the data for training and 
    # we generate input/output pairs for the training, validation, and test sets. 
    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    return transformed_X, sc

def trainshred(load_X, sensor_locations, lags):

    [n,m] = load_X.shape
    num_sensors = len(sensor_locations)

    train_indices, valid_indices, test_indices = partition_data(load_X, n, lags)
    transformed_X, sc = transform_data(load_X, train_indices)

    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    shred = models.SHRED(num_sensors, m, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS,
                            l1=L1_REGULARIZATION, l2=L2_REGULARIZATION, dropout=DROPOUT_RATE).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset,
                                    batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                                    lr=LEARNING_RATE, verbose=True, patience=PATIENCE)

    #Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

    return test_recons, test_ground_truth, shred, test_indices
   

def testshred(shred, hold_X, sensor_locations, lags):

    n,m = hold_X.shape
    num_sensors = len(sensor_locations)

    sc = MinMaxScaler()
    sc = sc.fit(hold_X)
    transformed_X_hold = sc.transform(hold_X)

    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X_hold[i:i+lags, sensor_locations]

    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #all_data_in = transformed_X_hold[:,newaxis,sensor_locations]
    test_data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)

    test_data_out = torch.tensor(transformed_X_hold[lags:,:], dtype=torch.float32).to(device)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    test_outputs = []

    batch_size = 512  # Adjust the batch size as needed
    for i in range(0, len(test_dataset.X), batch_size):
        batch = test_dataset.X[i:i+batch_size]
        with torch.no_grad():
            output = shred(batch)
            test_outputs.append(output)

    test_outall_outputs = torch.cat(test_outputs, dim=0)

    test_recons = sc.inverse_transform(test_outall_outputs.detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

    return test_recons, test_ground_truth