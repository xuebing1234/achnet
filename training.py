import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from models.conv2seq import Decoder as Conv2Seq
from models.con2seq_v2 import Decoder as Conv2SeqV2
import matplotlib.pyplot as plt
import pickle as pkl


def apply_random_mask(batch_data, no_mask_prob=0.5):
    """
    Apply a random mask to a batch of 1D crystal response data.

    Parameters:
        batch_data (torch.Tensor): Input tensor of shape (batch_size, response_dim).
        no_mask_prob (float): Probability of not applying a mask to a sample.

    Returns:
        torch.Tensor: Masked batch data.
        torch.Tensor: The mask applied (0 or 1 values).
    """
    batch_size, response_dim = batch_data.shape

    # Step 1: Decide for each sample whether to apply masking
    random_decision = torch.rand(batch_size)  # Random values in [0, 1]
    apply_mask = random_decision > no_mask_prob  # Masking decision (True/False)

    # Step 2: Initialize a mask tensor (1 for all samples initially)
    mask = torch.ones_like(batch_data)

    for i in range(batch_size):
        if apply_mask[i]:  # If masking is applied to this sample
            # Step 3: Generate a random masking ratio
            # masking_ratio = torch.rand(1).item() * 0.5 + 0.05  # Random value in [0, 1]
            masking_ratio = 0.2
            num_masked = int(masking_ratio * response_dim)  # Number of elements to mask

            # Step 4: Randomly select indices to mask
            mask_indices = torch.randperm(response_dim)[:num_masked]
            mask[i, mask_indices] = 0  # Set mask values to 0 for selected indices

    # Step 5: Apply the mask to the input batch
    masked_data = batch_data * mask

    return masked_data, mask


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


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        # return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class IntegrationLossScheduler:
    def __init__(self, start_weight=0, max_weight=0.01, warmup_epochs=1000):
        """
        Custom scheduler for integration loss weight.

        Parameters:
            start_weight (float): Initial weight for the integration loss.
            max_weight (float): Maximum weight for the integration loss.
            warmup_epochs (int): Number of epochs over which to increase the weight.
        """
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs

    def get_weight(self, epoch):
        """
        Calculate the weight for the current epoch.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Integration loss weight for this epoch.
        """
        if epoch >= self.warmup_epochs:
            return self.max_weight
        return self.start_weight + (self.max_weight - self.start_weight) * (epoch / self.warmup_epochs)
        # return self.start_weight


class CalculateMSE():
    def __init__(self, net, n_epochs, batch_size):
        super().__init__()
        self.net = net
        # initialize some constants
        self.batch_size = batch_size
        self.learning_rate = 4e-4
        self.n_epochs = n_epochs
        self.net.apply(self.weights_init)

    def weights_init(self, layer):
        if type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight)

    def get_mse(self, train_set, val_set):
        """
        train_set = torch.utils.data.TensorDataset(
            torch.Tensor(train_data),
            torch.Tensor(train_label))
        val_set = torch.utils.data.TensorDataset(
            torch.Tensor(test_data),
            torch.Tensor(test_label))
        """
        model_save = 'model_save/conv2seq_no_mask_hsirs.pt'
        val_loss_min = 100000
        loader_args = dict(batch_size=self.batch_size)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

        tloss = []
        vloss = []
        criterion = XSigmoidLoss()
        default_criterion = nn.MSELoss()
        # add weight decay
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=0)  # weight_decay=5e-4
        # optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) # weight_decay=0
        scheduler = IntegrationLossScheduler()

        with tqdm(range(0, self.n_epochs)) as pbar:
            for epoch in pbar:
                epoch_train_loss = []
                self.net.train()
                for i, data in enumerate(train_loader, 0):
                    inputs, label = data
                    inputs_mask, _ = apply_random_mask(inputs)
                    inputs = inputs.float().to(self.net.device)
                    inputs_mask = inputs_mask.float().to(self.net.device)
                    # y_pred = self.net(inputs.float().to(self.net.device))
                    y_pred, response_recon = self.net(inputs)
                    recon_loss = criterion(y_pred, label.float().to(self.net.device))
                    integration_loss = default_criterion(response_recon, inputs)
                    optimizer.zero_grad()

                    weight = scheduler.get_weight(epoch)
                    loss = recon_loss + weight * integration_loss
                    # if epoch >= 5:
                    #     loss = recon_loss + weight * integration_loss
                    # else:
                    #     loss = recon_loss
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(default_criterion(y_pred, label.to(self.net.device)).item())
                tloss.append(np.mean(epoch_train_loss))
                self.net.eval()
                epoch_loss = []
                for i, data in enumerate(val_loader, 0):
                    with torch.no_grad():
                        inputs1, label1 = data
                        inputs1 = inputs1.float().to(self.net.device)
                        # y_pred1 = self.net(inputs1.float().to(self.net.device))
                        y_pred1, response_recon = self.net(inputs1)
                        loss1 = default_criterion(y_pred1, label1.to(self.net.device))
                        epoch_loss.append(loss1.item())
                vloss.append(np.mean(epoch_loss))
                if vloss[-1] < val_loss_min:
                    val_loss_min = vloss[-1]
                    torch.save(self.net.state_dict(), model_save)

                pbar.set_postfix({'EPOCH': epoch,
                                  'tr_loss': tloss[-1],
                                  'val_loss': vloss[-1]})

        self.net.load_state_dict(torch.load(model_save))
        print("Minimum val loss:{}".format(val_loss_min))
        return self.net

    def get_mse_cv(self, train_folds, test_folds, response, spectra, device, params):
        # model_save = 'model_save/conv2seq_int_data_2.pt'

        loader_args = dict(batch_size=self.batch_size)
        for ind_cur, (train, test) in enumerate(zip(train_folds, test_folds)):
            """
            train_X = response[train]
            train_Y = spectra[train]
            test_X = response[test]
            test_Y = spectra[test]
            train_X = torch.tensor(train_X)
            train_Y = torch.tensor(train_Y)
            test_X = torch.tensor(test_X)
            test_Y = torch.tensor(test_Y)
            """
            # train_data, test_data, noise_ratio = add_noise(train[0], test[0], a=0.0508, std = 0.05,
            #                                                sequence_length=64, noise_seed=i)
            train_X = torch.tensor(train[0])
            train_Y = torch.tensor(train[1])
            test_X = torch.tensor(test[0])
            test_Y = torch.tensor(test[1])

            train_set = torch.utils.data.TensorDataset(train_X, train_Y)
            val_set = torch.utils.data.TensorDataset(test_X, test_Y)

            train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
            val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
            model_save = 'model_save/conv2seq_data_2_scheduler_long_{}.pt'.format(ind_cur)
            val_loss_min = 100000
            # mdl_cur = Conv2Seq(device, params)
            mdl_cur = Conv2SeqV2(device, params)
            mdl_cur.to(device)
            tloss = []
            vloss = []
            criterion = XSigmoidLoss()
            default_criterion = nn.MSELoss()
            # add weight decay
            self.learning_rate = 4e-4
            optimizer = optim.Adam(mdl_cur.parameters(), lr=self.learning_rate, weight_decay=0.0)  # weight_decay=5e-4
            # optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) # weight_decay=0
            scheduler = IntegrationLossScheduler()

            with tqdm(range(0, self.n_epochs)) as pbar:
                for epoch in pbar:
                    epoch_train_loss = []
                    mdl_cur.train()
                    # if epoch == 3000:
                    #     self.learning_rate = 1e-5
                    #     optimizer = optim.Adam(mdl_cur.parameters(), lr=self.learning_rate, weight_decay=0.0)
                    # elif epoch == 1000:
                    #     self.learning_rate = 1e-4
                    #     optimizer = optim.Adam(mdl_cur.parameters(), lr=self.learning_rate, weight_decay=0.0)
                    for i, data in enumerate(train_loader, 0):
                        inputs, label = data
                        inputs_mask, _ = apply_random_mask(inputs)
                        # inputs = inputs.float().to(self.net.device)
                        inputs_mask = inputs_mask.float().to(mdl_cur.device)
                        inputs = inputs.float().to(mdl_cur.device)
                        # y_pred = self.net(inputs.float().to(self.net.device))
                        y_pred, response_recon = mdl_cur(inputs_mask)
                        recon_loss = criterion(y_pred, label.float().to(mdl_cur.device))
                        # recon_loss = default_criterion(y_pred, label.float().to(mdl_cur.device))
                        integration_loss = default_criterion(response_recon, inputs)
                        optimizer.zero_grad()

                        weight = scheduler.get_weight(epoch)
                        loss = recon_loss + weight * integration_loss
                        loss.backward()
                        optimizer.step()
                        epoch_train_loss.append(default_criterion(y_pred, label.to(mdl_cur.device)).item())
                    tloss.append(np.mean(epoch_train_loss))
                    mdl_cur.eval()
                    epoch_loss = []
                    for i, data in enumerate(val_loader, 0):
                        with torch.no_grad():
                            inputs1, label1 = data
                            inputs1 = inputs1.float().to(mdl_cur.device)
                            # y_pred1 = self.net(inputs1.float().to(self.net.device))
                            y_pred1, response_recon = mdl_cur(inputs1)
                            loss1 = default_criterion(y_pred1, label1.to(mdl_cur.device))
                            epoch_loss.append(loss1.item())
                    vloss.append(np.mean(epoch_loss))
                    if vloss[-1] < val_loss_min:
                        val_loss_min = vloss[-1]
                        torch.save(mdl_cur.state_dict(), model_save)

                    pbar.set_postfix({'EPOCH': epoch,
                                      'tr_loss': tloss[-1],
                                      'val_loss': vloss[-1]})
            print("Minimum val loss:{}".format(val_loss_min))
            epochs = range(len(tloss)-100)


            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, np.array(tloss)[100:], color='#40833b', label='Training Loss')
            plt.plot(epochs, np.array(vloss)[100:], color='#9131a2', label='Validation Loss')

            # Add title and labels
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            # Add a legend to differentiate the curves
            plt.legend()

            # Display the plot
            plt.show()
            with open("loss_backbone{}.pkl".format(ind_cur), 'wb') as f:
                pkl.dump([tloss, vloss], f)
        # self.net.load_state_dict(torch.load(model_save))

        return self.net
