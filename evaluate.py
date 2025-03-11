from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import kendalltau, spearmanr
import dcor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)


def eval_corr(idx, true_spectra, recon_spectra):
    cur_true = true_spectra[idx]
    cur_recon = recon_spectra[idx]
    spearman_corr, _ = spearmanr(cur_true, cur_recon)
    distance_corr = dcor.distance_correlation(cur_true, cur_recon)
    dtw_cur, _ = fastdtw(cur_true.reshape(-1, 1), cur_recon.reshape(-1, 1), dist=euclidean)
    return spearman_corr, distance_corr, dtw_cur


def eval_recon(true_spectra, recon_spectra):
    mse = np.mean((recon_spectra - true_spectra) ** 2)

    reconstruction_error = np.mean((recon_spectra - true_spectra) ** 2, axis=-1)
    median_error = np.median(reconstruction_error)
    error_90_percentile = np.percentile(reconstruction_error, 90)

    spearman_corr_list = []
    distance_corr_list = []
    dtw_list = []
    if len(true_spectra.shape) > 2:
        n_wavelength = true_spectra.shape[-1]
        true_spectra = np.reshape(true_spectra, (-1, n_wavelength))
        recon_spectra = np.reshape(true_spectra, (-1, n_wavelength))

    n_sample = len(true_spectra)
    n_workers = 48

    """
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        with tqdm(range(0, n_sample)) as pbar:
            results = list(executor.map(
                eval_corr,
                pbar,  # The indices
                [true_spectra]*n_sample,  # Broadcast true_spectra
                [recon_spectra]*n_sample  # Broadcast recon_spectra
            ))

    for spearman_corr, distance_corr, dtw_cur in results:
        spearman_corr_list.append(spearman_corr)
        distance_corr_list.append(distance_corr)
        dtw_list.append(dtw_cur)
    """
    with tqdm(range(0, n_sample)) as pbar:
        for i in pbar:
            cur_true = true_spectra[i]
            cur_recon = recon_spectra[i]
            spearman_corr, _ = spearmanr(cur_true, cur_recon)
            distance_corr = dcor.distance_correlation(cur_true, cur_recon)
            dtw_cur, _ = fastdtw(cur_true.reshape(-1, 1), cur_recon.reshape(-1, 1), dist=euclidean)

            spearman_corr_list.append(spearman_corr)
            distance_corr_list.append(distance_corr)
            dtw_list.append(dtw_cur)

    spearman_corr_overall = np.nanmean(spearman_corr_list)
    distance_corr_overall = np.nanmean(distance_corr_list)
    dtw_overall = np.nanmean(dtw_list)

    print("The overall MSE: {}".format(mse))
    print("The median MSE: {}".format(median_error))
    print("The 90 percent error: {}".format(error_90_percentile))
    print("The DTW: {}".format(dtw_overall))
    print("The spearman correlation: {}".format(spearman_corr_overall))
    print("The distance correlation: {}".format(distance_corr_overall))
