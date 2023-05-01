from xml.sax.xmlreader import InputSource
import ecg_reader
import utils
import losses
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras import mixed_precision
from numpy.random import randint
import os
import math
from dataclasses import dataclass
import sys
from tqdm import tqdm


@dataclass
class EpochSample:
    ecg: any  # can be tensor or nparray
    codes: any # How many samples to produce after each epoch for the plots.
    disc_score: any


class CGAN:
    BUFFER_SIZE = 120000  # The Buffer size for shuffling the dataset. MNIST has 60000 pictures
    # How many images are contained in each batch. Higher numbers = faster, but more VRAM needed
    BATCH_SIZE = 96
    CHECKPOINT_PATH = "models/{}.ckpt"
    N_SAMPLES = 64
    TINY = 1e-8  # used in the losses to avoid log(0)

    def __init__(self, latent_dim, n_epochs, learning_rate_gen, learning_rate_disc, dropout_perc, max_filter, run_name):
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.dropout_perc = dropout_perc
        self.max_filter = max_filter
        self.run_name = run_name

        # Turn on 16 bit floating point precision which helps with training speeds
        # mixed means the computations are fp16, and the variables are stored in fp32
        # mixed_precision.set_global_policy('mixed_float16')

        data = np.load("../data/synthetic/beats3.npy")
        info = np.load("../data/synthetic/beats3_labels.npy")

        one_hot_peak_locations = []

        info = utils.peak_location_to_one_hot(info, 1024)
        info = info.astype(np.float32)
        indices = np.random.randint(0, data.shape[0], size=60000)
        self.train_data = data[indices, :]
        self.train_labels = info[indices, :]

        self.series_length = len(self.train_data[0])
        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_gen)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_disc)
        self.auxillary_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_disc)
        self.seeds = tf.random.normal(
            [int(CGAN.N_SAMPLES / 2), self.latent_dim])
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data, self.train_labels)).shuffle(CGAN.BUFFER_SIZE).batch(CGAN.BATCH_SIZE)

        self._make_generator_model()
        self._make_discriminator_model()


    def _make_generator_model(self):
        """ 
        This function defines the generator using TFs functional API.
        It upsamples the starting latent space vector through a series
        of transpose convolutions. The Rpeak location input gets samples down
        through convolution and concatted to the upsamples latent space vector.
        output dimensions are (1024, 1)
        """
        noise_input = layers.Input(
            shape=(self.latent_dim), batch_size=CGAN.BATCH_SIZE)
        noise_up = layers.Dense(128 * 64, activation="swish")(noise_input)
        noise_reshape = layers.Reshape((128, 64))(noise_up)


        rpeak_input = layers.Input(shape=(1024,), batch_size=CGAN.BATCH_SIZE)
        rpeak_input = layers.Reshape((1024, 1))(rpeak_input)

        rpeak_down1 = layers.Conv1D(
            128, 16, strides=2, padding="same", activation="swish")(rpeak_input)
        rpeak_down2 = layers.Conv1D(
            128, 16, strides=2, padding="same", activation="swish")(rpeak_down1)
        rpeak_down3 = layers.Conv1D(
            64, 16, strides=2, padding="same", activation="swish")(rpeak_down2)


        concatted = layers.Concatenate()(
            [noise_reshape, rpeak_down3])  # (Batch, 128, 128)

        batch_norm_1 = layers.BatchNormalization()(concatted)
        up_1 = layers.Conv1DTranspose(
            128, 170, strides=1, padding="same", activation="LeakyReLU")(batch_norm_1)

        batch_norm_2 = layers.BatchNormalization()(up_1)
        up_2 = layers.Conv1DTranspose(
            64, 170, strides=2, padding="same", activation="LeakyReLU")(batch_norm_2)

        batch_norm_3 = layers.BatchNormalization()(up_2)
        up_3 = layers.Conv1DTranspose(
            32, 25, strides=2, padding="same", activation="LeakyReLU")(batch_norm_3)

        batch_norm_4 = layers.BatchNormalization()(up_3)
        up_4 = layers.Conv1DTranspose(
            128, 170, strides=2, padding="same", activation="LeakyReLU")(batch_norm_4)

        conv_out = layers.Conv1DTranspose(
            1, 51, 1, padding="same", activation="LeakyReLU")(up_4)


        self.generator = models.Model(
            inputs=[noise_input, rpeak_input], outputs=conv_out)
        self.generator.summary()
        return self.generator

    def _make_discriminator_model(self):
        """
        This function defines the discriminator using TFs functional API.
        It takes the ECG of shape (1024, 1) as well as the rpeak codes,
        and applies a series of convolutional and pooling layers. At the end, 
        the output gets flattened and a final dense
        layer outputs a number between 0 and 1.
        """
        ecg_input = layers.Input(shape=(self.series_length, 1))
        rpeak_input = layers.Input(shape=(self.series_length, 1))

        combined = layers.Concatenate()([ecg_input, rpeak_input])

        conv1 = layers.Conv1D(self.max_filter, int(
            self.series_length/128), strides=2, padding="same")(combined)
        lkyReLU1 = layers.LeakyReLU()(conv1)

        drp1 = layers.Dropout(self.dropout_perc)(lkyReLU1)
        conv2 = layers.Conv1D(
            self.max_filter * 2, int(self.series_length/64), strides=2, padding='same')(drp1)
        lkyReLU2 = layers.LeakyReLU()(conv2)


        drp2 = layers.Dropout(self.dropout_perc)(lkyReLU2)
        conv3 = layers.Conv1D(self.max_filter, int(
            self.series_length / 16), strides=2, padding='same')(drp2)
        lkyReLU3 = layers.LeakyReLU()(conv3)


        drp3 = layers.Dropout(self.dropout_perc)(lkyReLU3)
        conv4 = layers.Conv1D(
            self.max_filter / 2, int(self.series_length / 16), strides=4, padding='same')(drp3)
        lkyReLU4 = layers.LeakyReLU()(conv4)


        fltn = layers.Flatten()(lkyReLU4)
        out_dense = layers.Dense(1, activation="sigmoid")(fltn)

        self.discriminator = models.Model(
            inputs=[ecg_input, rpeak_input], outputs=out_dense)
        self.discriminator.summary()

        return self.discriminator

    def _generate_epoch_samples(self):
        """
        Calls the generator model to generate ECGs. Used for plotting the progress at the end of each epoch.
        The codes are generated by drawing from the training dataset rpeak distribution.
        """
        noise = tf.random.normal([int(CGAN.N_SAMPLES), self.latent_dim])

        codes = self.get_codes(CGAN.N_SAMPLES)
        # codes = np.array([utils.peak_location_to_one_hot(i)
        #                  for i in codes])

        indices = np.random.randint(0, self.train_data.shape[0], size=64)
        true_ecgs = self.train_data[indices, :]
        true_codes = self.train_labels[indices, :]
        true_score = self.discriminator([true_ecgs, true_codes])
        #true_pred = self.auxillary(true_ecgs)
        #x_input = tf.concat([self.seeds, noise], 0)
        #input = self._concat_inputs([noise, codes])
        #input = noise
        X = self.generator([noise, codes])
        gen_score = self.discriminator([X, codes])
        #gen_pred = self.auxillary(X)

        sample_real = EpochSample(true_ecgs, true_codes, true_score)
        sample_gen = EpochSample(X, codes, gen_score)
        return sample_real, sample_gen

    def save_output(self, output, filename):
        """
        Prints a tensor or numpy array into the provided output file.
        Useful for debugging.
        """
        if type(output[0]) == tf.Tensor:
            numpied = [x.numpy() for x in tf.reshape(output, [-1])]
        else:
            numpied = output
        print([x.shape for x in numpied])
        np.savetxt("output.txt", numpied, delimiter=";")

    def train(self):
        """
        This function defines the main train loop.
        It goes through the training data, calls the train step for each batch
        and prints the current progress after each epoch.
        """
        for epoch in range(self.n_epochs):
            pbar_disc = tqdm(range(len(self.train_dataset)),
                             desc=f"Epoch {epoch}")

            real_out = []
            fake_out = []
            aux_losses = []
            for (i, batch) in zip(pbar_disc, self.train_dataset):
                disc_loss, gen_loss = self._train_step(
                    batch, train_discriminator=True)
                disc_loss, gen_loss = self._train_step(
                    batch, train_discriminator=False)
                pbar_disc.set_postfix(
                    {"Disc Loss: ": f"{disc_loss:.3f}", "Gen Loss: ": f"{gen_loss: .3f}"})


            real_samples, gen_samples = self._generate_epoch_samples()
            if not os.path.exists(f"images/{self.run_name}/{epoch}/"):
                os.makedirs(f"images/{self.run_name}/{epoch}/")

            for i in range(64):
                ecg = gen_samples.ecg[i]
                codes = gen_samples.codes[i]
                codes = tf.reshape(codes, (1, 1024))
                ecg_input = tf.reshape(ecg, (1, 1024))
                disc_score = self.discriminator([ecg_input, codes])
                disc_score = np.round(disc_score.numpy(), decimals=2)
                utils.save_single_ecg_plot(ecg, gen_samples.codes[i], f"images/{self.run_name}/{epoch}/{i}",
                                           f"Generator output after {epoch} epochs. Discriminator Scoe: {str(disc_score).lstrip('[').rstrip(']')}")
            utils.save_plot(f"images/{self.run_name}/{epoch}/true", real_samples.ecg, real_samples.disc_score,
                            real_samples.codes, f"True ECGs and discriminator score after {epoch} epochs", epoch)

            utils.save_plot(f"images/{self.run_name}/{epoch}/generated", gen_samples.ecg, gen_samples.disc_score,
                            gen_samples.codes, f"Generator output after {epoch} epochs and discriminator score", epoch)

            #self.save_code_plot(epoch, ecgs2, codes2)

        # save models
        self.generator.save_weights(
            CGAN.CHECKPOINT_PATH.format("generator"))
        self.discriminator.save_weights(
            CGAN.CHECKPOINT_PATH.format("discriminator"))

    def _concat_inputs(self, input):
        concat_input = layers.Concatenate()(input)
        return concat_input

    def get_codes(self, amount):
        index = randint(0, len(self.train_labels), size=amount)
        orig_codes = self.train_labels[index]
        return orig_codes

    @tf.function
    def _train_step(self, real_input, train_discriminator=True):
        """
        The actual train step. For each batch the generator generates a fake batch
        then both get fed into the discriminator. Calculates the loss and gradients
        and applies them to the models.
        """
        real_ecg = real_input[0]
        real_codes = real_input[1]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as aux_tape, tf.GradientTape() as core_tape:
            noise = tf.random.normal([CGAN.BATCH_SIZE, self.latent_dim])
            codes = self.get_codes(CGAN.BATCH_SIZE)

            generator_input = [noise, codes]
            generated_ecg = self.generator(generator_input, training=True)

            real_output = self.discriminator(
                [real_ecg, real_codes], training=True)
            fake_output = self.discriminator(
                [generated_ecg, codes], training=True)

            disc_loss = -1*losses.discriminator_loss(real_output, fake_output)


            gradients_of_discriminator = core_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)
            if train_discriminator:
                self.discriminator_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, self.discriminator.trainable_variables))



            gen_loss = -1*disc_loss
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))

            return disc_loss, gen_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="default",
                        help="The name of the folder to store all outputs")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Defines some hyperparameters. Change here if you want to run with different settings.
    """
    args = parse_args()
    if not os.path.exists("images/" + args.name):
        os.makedirs("images/" + args.name)
    gan = CGAN(latent_dim=50, n_epochs=400, learning_rate_gen=1e-4,
               learning_rate_disc=5e-6, dropout_perc=0.3, max_filter=128, run_name=args.name)
    gan.train()
