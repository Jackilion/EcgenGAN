"""
Defines some helper functions for the cGAN.
"""

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from typing import List


def peak_location_to_one_hot_128(peaks) -> List:
    '''
    converts absolute peak location between 1 and 1024 into
    one hot encoded segments of length 8
    e.g. a single peak at sample point 15 would become encodes like so:
    [0, 1, 0 , 0, ...]
    '''
    peak_one_hot = [0.0 for i in range(128)]
    for peak in peaks:
        if (peak % 8 == 0 and peak != 0):
            peak -= 1
        loc = int(peak) // 8
        peak_one_hot[loc] = 1.0
    return np.asarray(peak_one_hot, dtype=np.float32)


def peak_location_to_one_hot(peaks, series_length) -> List:
    B, N = peaks.shape
    zeros = np.zeros(B * series_length)

    non_zero_multiidx = np.array(
        (np.repeat(
            np.arange(B),
            N
        ),
            np.reshape(peaks, (-1,))
        )
    )
    non_zero_inds = np.ravel_multi_index(
        non_zero_multiidx, (B, series_length), mode='clip')
    zeros[non_zero_inds] = 1.0
    delta_series = np.reshape(zeros, (-1, series_length))
    delta_series[:, 0] = 0.0

    return delta_series


def one_hot_to_peak_location(code):
    return np.where(code == 1.0)


def one_hot_peak_location_to_segment(code):
    numbers = []
    for i in range(len(code)):
        if tf.math.round(code[i]) == 1:
            numbers.append(i + 1)
    return numbers


def logits_to_one_hot(code):
    #numpied = code.numpy()
    #indices = np.argpartition(numpied, -3)[-3:]
    top_3 = tf.math.top_k(code, k=3)
    indices = top_3.indices
    # print(indices)
    #indices = tf.cast(indices, tf.int32)
    #new_code = [0.0 for i in range(len(code))]
    new_code = tf.zeros_like(code)
    updates = tf.ones_like(indices, dtype=tf.float32)
    new_code = tf.tensor_scatter_nd_update(
        new_code, indices[..., None], updates)
    return new_code
    for i in indices:
        print(i)
        #new_code[i] = 1.0
        #new_code[indices, :] = 1.0
    return new_code


def summarize_epoch(epoch, real_out, fake_out, aux_loss) -> str:
    summary = ""
    summary += "Epoch number: {} \n".format(epoch)

    real_avg = np.average(
        [tf.math.reduce_mean(out).numpy() for out in real_out])
    fake_avg = np.average(
        [tf.math.reduce_mean(out).numpy() for out in fake_out])
    aux_loss_avg = np.average(
        [tf.math.reduce_mean(out).numpy() for out in aux_loss])

    summary += "Average prediction of real ecg: {} \n".format(real_avg)
    summary += "Average prediction of fake ecg: {} \n".format(fake_avg)
    summary += "Aux loss: {} \n".format(aux_loss_avg)
    return summary


# test = peak_location_to_one_hot([0, 16, 128, 500, 753, 1023])
# # print(test)
# test2 = one_hot_peak_location_to_segment(test)
# test2 = one_hot_peak_location_to_segment([0.5, 0.7, 0.3, 1.0, 0.0])
# print(test2)
# print(test2)

def save_single_ecg_plot(ecg, codes, path, title=None):
    # print(ecg.shape)
    # print(ecg)
    peak_locations = one_hot_to_peak_location(codes)

    plt.plot(ecg)
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title(f"{title}")
    plt.vlines(peak_locations, ymin = -0.5, ymax = 1.0, colors="red", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_plot(path, ecgs, disc_scores, codes, title, epoch=None) -> None:
    # plt.tight_layout()
    '''
    Takes 50 ecgs, their discriminator scores and codes
    and displays them on a figure
    '''
    # print(disc_codes[0].numpy())
    # converted = one_hot_peak_location_to_segment(disc_codes[0])
    # print(converted)

    x = tf.reshape(disc_scores, [-1])
    sort_idx = np.argsort(x)
    fig = plt.figure(figsize=(8, 8))
    for i in range(64):
        #ax = axs.flat[i]
        ax = plt.subplot(8, 8, i+1)
        labels=codes[sort_idx[i]]
        labels_location = one_hot_to_peak_location(labels)
        #ax = fig.subplot(5, 10, i+1)
        ax.plot(ecgs[sort_idx[i]])
        ax.axis('off')
        ax.vlines(labels_location, ymin = -0.5, ymax = 1.0, colors="red", alpha=0.2)
        score_text = str(np.round(x[sort_idx[i]].numpy(), decimals=2))
        ax.text(0.5, -0.1, score_text, ha="center",
                size=5, transform=ax.transAxes)
        #codes_encoded = one_hot_peak_location_to_segment(codes[sort_idx[i]])
        #disc_codes_3_maxed = logits_to_one_hot(disc_codes[sort_idx[i]])
        # disc_codes_encoded = one_hot_peak_location_to_segment(
        #    disc_codes_3_maxed)
        codes_text = ", ".join([str(code) for code in codes])

        #disc_codes_text = ", ".join([str(code) for code in disc_codes_encoded])
        # if len(disc_codes_encoded) > 5:
        #    disc_codes_text = "too many ({})".format(len(disc_codes_encoded))
        # ax.text(0.5, -0.3, codes_text, ha="center",
        #        size=5, transform=ax.transAxes)
        # ax.text(0.5, -0.5, disc_codes_text, ha="center",
        #        size=5, transform=ax.transAxes)
    plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_summary(summary) -> None:
    if not os.path.exists("metrics/"):
        os.makedirs("metrics")
    f = open("metrics/output.txt", "a")
    f.write(summary)
    f.close()


# def save_plots()

# test = peak_location_to_one_hot([0, 16, 128, 500, 753, 1023])
# print(test)
