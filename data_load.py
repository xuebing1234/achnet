from torch.utils.data import DataLoader, random_split
import numpy as np
import scipy.io as io
import pandas as pd
import torch
import pickle as pkl


def data_process():
    num_of_filter = 64
    num_of_wavelength = 251
    num_sample = 2056

    response = pd.read_csv("data/our_data/response.csv", header=None)  # input X
    response = np.array(response)
    # response = response[0:nu, 0:num_of_filter]
    response = torch.FloatTensor(response.tolist())
    response = response.view(num_sample, 1, num_of_filter)

    spectra = pd.read_csv("data/our_data/spectra.csv", header=None)
    spectra = np.array(spectra)
    spectra = spectra[0:num_sample, 0:num_of_wavelength]
    spectra = torch.FloatTensor(spectra.tolist())
    spectra = spectra.view(num_sample, 1, num_of_wavelength)

    return response, spectra


def torch_data_loader(response, spectra, num_of_wavelength=251):

    size = num_of_wavelength
    InputX = torch.tensor(response)
    LabelY = torch.tensor(spectra)
    myDataset = torch.utils.data.TensorDataset(InputX, LabelY)
    val_percent = 0.125
    test_percent = 0.125
    n_val = int(len(myDataset) * val_percent)
    n_test = int(len(myDataset) * test_percent)
    n_train = len(myDataset) - n_val - n_test
    train_set, val_set, test_set = random_split(myDataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(0))
    return train_set, val_set, test_set


def add_noise(inputs, inputs2, a=0.02, std=0.02, sequence_length=64, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
    noise = np.random.normal(0, std, size=(inputs.shape[0], sequence_length)).astype(np.float32)
    noise2 = np.random.normal(0, std, size=(inputs2.shape[0], sequence_length)).astype(np.float32)
    # noise = poisson.rvs(mu, size=(inputs.shape[0], sequence_length)).astype(np.float32)

    nsd = a * np.random.poisson(inputs / a).astype(np.float32)
    nsd2 = a * np.random.poisson(inputs2 / a).astype(np.float32)

    # Calculate the absolute error between nsd and original inputs
    absolute_error = np.abs(nsd - inputs)
    absolute_error2 = np.abs(nsd2 - inputs2)
    # Sum up the absolute errors
    total_error = np.sum(absolute_error)
    total_error2 = np.sum(absolute_error2)
    # Sum of the original data
    total_original = np.sum(inputs)
    total_original2 = np.sum(inputs2)

    # Calculate the noise ratio
    noise_ratio = (total_error + total_error2) / (total_original + total_original2)

    return nsd, nsd2, noise_ratio


def data_process_data2():
    response = pd.read_csv("data/our_data/data_1/response.csv", header=None).values  # input X
    spectra = pd.read_csv("data/our_data/data_1/spectra.csv", header=None).values

    with open("data/our_data/data_2_resampled_2x5_cross_validation.pickle", 'rb') as f:
        b = pkl.load(f)

    train_fold = b[0]
    val_fold = b[1]
    return train_fold, val_fold, response, spectra


def data_process_hsirs():
    def min_max_normalization(data):
        normalized_data = (data - data.min(axis=1, keepdims=True)) / (
                    data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-8)
        return normalized_data
    features = np.load("data/hsirs/dataset2_X.npy")
    spectra = np.load("data/hsirs/dataset2_Y.npy")
    
    # features = np.load("data/hsirs/dataset2_X.npy")
    # spectra = np.load("data/hsirs/dataset2_Y.npy")

    features = min_max_normalization(features[:10000000, :])
    spectra = min_max_normalization(spectra[:10000000, :])
    
    return features, spectra
