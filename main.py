import argparse
import data_load
from baseline import run_baseline
import evaluate
from training import CalculateMSE
from models.conv2seq import Decoder as Conv2Seq
from models.con2seq_v2 import Decoder as ACHNet
from models.dnn import DNN
from models.unet import UNet
from models.rnn import GRU, LSTM, Seq2SeqGRU
from models.wnn import WaveNet
from models.transformer import LightweightTransformer
import torch
import numpy as np
from torch.utils.data import DataLoader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def ista_main(args):
    response_all, spectra_all = data_load.data_process()
    # response_all, spectra_all = data_load.data_process_hsirs()
    test_spectra, reconstructed_spectra = run_baseline(args, response_all, spectra_all)
    evaluate.eval_recon(test_spectra, reconstructed_spectra)


def test_cv(device, mdl, val_fold, response, spectra):
    spectra_list_all = []
    recon_list_all = []
    for i, test_data in enumerate(val_fold):
        model_save = 'model_save/conv2seq_data_2_scheduler_{}.pt'.format(i)
        mdl.load_state_dict(torch.load(model_save))
        spectra_list = []
        recon_list = []
        # test_X = torch.tensor(response[test_data])
        # test_Y = torch.tensor(spectra[test_data])
        test_X = torch.tensor(test_data[0])
        test_Y = torch.tensor(test_data[1])
        test_set = torch.utils.data.TensorDataset(test_X, test_Y)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=128)
        mdl.eval()
        for j, data in enumerate(test_loader, 0):
            inputs1, label1 = data
            # y_pred1 = model(inputs1.float().to(device))
            with torch.no_grad():
                y_pred1, _ = mdl(inputs1.float().to(device))
                # y_pred1 = mdl(inputs1.float().to(device))
            # if i == 8:
            spectra_list.append(label1.float().detach().cpu().numpy())
            recon_list.append(y_pred1.detach().cpu().numpy())
        spectra_list_all.append(np.concatenate(spectra_list, axis=0))
        recon_list_all.append(np.concatenate(recon_list, axis=0))
        # spectra_list_all.append(spectra_list)
        # recon_list_all.append(recon_list)

    spectra_list_all = np.concatenate(spectra_list_all, axis=0)
    recon_list_all = np.concatenate(recon_list_all, axis=0)
    # spectra_list_all = np.array(spectra_list_all)
    # recon_list_all = np.array(recon_list_all)
    return spectra_list_all, recon_list_all


def conv2seq_main(params):
    device = str('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')
    losses = []
    features, spectra = data_load.data_process_hsirs()
    train_data, val_data, test_set = data_load.torch_data_loader(features, spectra, num_of_wavelength=33)
    # Adding noise to the train and test data
    # train_data, test_data, noise_ratio = add_noise(train_data, test_data, a=0.0065, std=0.05, sequence_length=64,
    #                                                noise_seed=i)
    # print(noise_ratio)
    # noise_ratios.append(noise_ratio)
    # train_fold, test_fold, response, spectra = data_load.data_process_data2()
    # mdl = Conv2Seq(device, params)
    mdl = ACHNet(device, params)
    # mdl = DNN(device, params.num_of_filter, params.num_of_wavelength)
    # mdl = UNet(device, params.num_of_filter, params.num_of_wavelength)
    # mdl = Seq2SeqGRU(device, params)
    # mdl = LightweightTransformer(device, params)

    mdl.to(device)
    mse_calculator = CalculateMSE(mdl, params.epochs, params.batch_size)
    model = mse_calculator.get_mse(train_data, val_data)
    # model = mse_calculator.get_mse_cv(train_fold, test_fold, response, spectra, device, params)
    # model.to(device)
    """
    model.eval()

    test_loader = DataLoader(test_set, shuffle=True, batch_size=256)
    spectra_list = []
    recon_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs1, label1 = data
            # y_pred1 = model(inputs1.float().to(device))
            y_pred1, _ = model(inputs1.float().to(device))
            spectra_list.append(label1.float().detach().cpu().numpy())
            recon_list.append(y_pred1.detach().cpu().numpy())

    spectra_list = np.concatenate(spectra_list, axis=0)
    recon_list = np.concatenate(recon_list, axis=0)
    """
    spectra_list, recon_list = test_cv(device, mdl, test_fold, response, spectra)
    evaluate.eval_recon(spectra_list, recon_list)


class RnnType:
    GRU = 1
    LSTM = 2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_filter', type=int, default=64)
    parser.add_argument('--num_of_wavelength', type=int, default=250)
    parser.add_argument('--chs_encoder', type=tuple, default=(1, 32, 64))
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--rnn_hidden_dim', type=int, default=32)
    parser.add_argument('--layer_width', type=int, default=2000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--sequence_length', type=int, default=64)
    parser.add_argument('--output_sequence_length', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--num_of_wavelength', type=int, default=251)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--rnn_type', default=RnnType.GRU)

    args = parser.parse_args()
    # ista_main(args)
    conv2seq_main(args)

