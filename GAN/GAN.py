#import ecg_reader
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from numpy.random import randint
import os
import math
from tqdm import tqdm
import sys


class GAN:
    BUFFER_SIZE = 120000  # The Buffer size for shuffling the dataset.
    BATCH_SIZE = 512  # How many images are contained in each batch. Higher numbers = faster, but more VRAM needed
    EVAL_BATCH_SIZE = 1024
    N_SAMPLES = 64   # How many samples to produce after each epoch for the plots.
    CHECKPOINT_PATH = "models/{}.ckpt"

    def __init__(self, latent_dim, n_epochs, learning_rate_gen, learning_rate_disc, dropout_perc, max_filter):
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.dropout_perc = dropout_perc
        self.max_filter = max_filter

        # Turn on 16 bit floating point precision which helps with training speeds
        # mixed means the computations are fp16, and the variables are stored in fp32
        # mixed_precision.set_global_policy('mixed_float16')

        #self.train_data = ecg_reader.load_data("../data/incartdb", 0, 75, 1)
        data = np.genfromtxt("../data/synthetic/beats3.csv", delimiter=",")
        #data = np.transpose(data)

        info = np.genfromtxt(
            "../data/synthetic/beats3_labels.csv", delimiter=",")
        normalized_codes = [
            [x / 1024.0 if x != -1.0 else x for x in y] for y in info]
        beats4_idx = []
        for i in range(len(normalized_codes)):
            if(normalized_codes[i].count(-1.0) == 0):
                beats4_idx.append(i)

        beats4_ecgs = [data[i] for i in beats4_idx]
        beats4_info = [normalized_codes[i] for i in beats4_idx]
        self.train_data = np.asarray(beats4_ecgs)
        self.train_labels = np.asarray(beats4_info)

        print("length: " + str(len(self.train_data[0])))
        self.series_length = len(self.train_data[0])
        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_gen)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_disc)
        self.seeds = tf.random.normal(
            [int(GAN.N_SAMPLES / 2), self.latent_dim])
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            self.train_data).shuffle(GAN.BUFFER_SIZE).batch(GAN.BATCH_SIZE)

        self._make_generator_model()
        self._make_discriminator_model()

    def _make_generator_model(self):
        """ 
        This function defines the generator using TFs sequential API.
        It upsamples the starting latent space vector through a series
        of transpose convolutions until it arrives at the desired
        output dimension of (1024, 1)
        """
        self.generator = tf.keras.Sequential()
        self.generator.add(layers.Dense(int(self.series_length / 8) *
                           self.max_filter, use_bias=False, input_shape=(self.latent_dim,)))
        self.generator.add(layers.LeakyReLU())

        self.generator.add(layers.Reshape(
            (int(self.series_length / 8), self.max_filter)))
        assert self.generator.output_shape == (
            None, self.series_length / 8, self.max_filter)

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(self.max_filter, int(
            self.series_length/6), strides=1, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        assert self.generator.output_shape == (
            None, int(self.series_length / 8), self.max_filter)

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(self.max_filter / 2),
                           int(self.series_length/6), strides=2, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        assert self.generator.output_shape == (
            None, int(self.series_length / 4), int(self.max_filter / 2))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(self.max_filter/4), int(
            self.series_length / 40), strides=2, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        assert self.generator.output_shape == (
            None, int(self.series_length / 2), int(self.max_filter / 4))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(
            self.max_filter/8), int(self.series_length/6), strides=2, padding="same", use_bias=True))
        self.generator.add(layers.LeakyReLU())
        assert self.generator.output_shape == (
            None, self.series_length, int(self.max_filter / 8))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(
            1, int(self.series_length / 20), strides=1, padding="same", use_bias=True))
        # self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.summary()

        assert self.generator.output_shape == (None, self.series_length, 1)
        return self.generator

    def _make_discriminator_model(self):
        """
        This function defines the discriminator using TFs sequential API.
        It takes the ECG of shape (1024, 1) and applies a series of convolutional
        and pooling layers. At the end, the output gets flattened and a final dense
        layer outputs a number between 0 and 1.
        """
        self.discriminator = tf.keras.Sequential()

        self.discriminator.add(layers.Conv1D(self.max_filter, int(
            self.series_length/128), strides=2, padding="same", input_shape=[self.series_length, 1]))
        self.discriminator.add(layers.LeakyReLU())
        assert self.discriminator.output_shape == (
            None, int(self.series_length / 2), self.max_filter)

        self.discriminator.add(layers.Dropout(self.dropout_perc))
        self.discriminator.add(layers.Conv1D(
            self.max_filter * 2, int(self.series_length/64), strides=2, padding='same'))
        self.discriminator.add(layers.LeakyReLU())
        assert self.discriminator.output_shape == (
            None, int(self.series_length / 4), self.max_filter * 2)

        self.discriminator.add(layers.Dropout(self.dropout_perc))
        self.discriminator.add(layers.Conv1D(self.max_filter, int(
            self.series_length / 16), strides=2, padding='same'))
        self.discriminator.add(layers.LeakyReLU())
        assert self.discriminator.output_shape == (
            None, int(self.series_length / 8), self.max_filter)

        self.discriminator.add(layers.Dropout(self.dropout_perc))
        self.discriminator.add(layers.Conv1D(
            self.max_filter / 2, int(self.series_length / 16), strides=4, padding='same'))
        self.discriminator.add(layers.LeakyReLU())
        assert self.discriminator.output_shape == (
            None, int(self.series_length / 32), 64)

        self.discriminator.add(layers.Flatten())
        self.discriminator.add(layers.Dense(1, activation="sigmoid"))
        self.discriminator.summary()
        return self.discriminator

    def _discriminator_loss(self, real_output, fake_output):
        """
        The loss function for the discriminator.
        """
        loss1 = 1*tf.keras.backend.mean(tf.math.log(real_output))
        loss2 = 1 * \
            tf.keras.backend.mean(tf.math.log(tf.math.subtract(
                tf.ones_like(fake_output), fake_output)))
        return loss1 + loss2

    def _generator_loss(self, fake_output):
        """
        The loss function for the generator
        """
        #fake_output = tf.sigmoid(fake_output)
        return tf.keras.backend.mean(tf.math.subtract(tf.ones_like(fake_output), fake_output))
        # WGAN:
        # return -1*tf.keras.backend.mean(fake_output)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def _generate_fake_samples(self, g_model):
        """
        Calls the generator model to generate ECGs. Used for plotting the progress at the end of each epoch
        """
        noise = tf.random.normal([int(GAN.N_SAMPLES / 2), self.latent_dim])
        x_input = tf.concat([self.seeds, noise], 0)
        X = g_model(x_input)
        return X

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
        epochs = []
        for epoch in range(self.n_epochs):
            real_out = []
            fake_out = []
            disc_loss = []
            gen_loss = []
            pbar = tqdm(range(len(self.train_dataset)), desc=f"Epoch {epoch}")
            for (_, batch) in zip(pbar, self.train_dataset):
                ro, fo, dl, gl = self._train_step(batch)
                real_out.append(ro)
                fake_out.append(fo)
                disc_loss.append(dl)
                gen_loss.append(gl)
                pbar.set_postfix(
                    {"Disc loss: ": dl.numpy(), "Gen loss: ": gl.numpy()})

            print("Epoch number: " + str(epoch))
            real_avg = np.average(
                [tf.math.reduce_mean(out).numpy() for out in real_out])
            fake_avg = np.average(
                [tf.math.reduce_mean(out).numpy() for out in fake_out])
            gen_loss_avg = np.average(
                [tf.math.reduce_mean(out).numpy() for out in gen_loss])
            disc_loss_avg = np.average(
                [tf.math.reduce_mean(out).numpy() for out in disc_loss])
            if np.isnan(real_avg) or np.isnan(fake_avg) or any(np.isnan(disc_loss)) or any(np.isnan(gen_loss)):
                self.save_output(real_out, "real_out")
                self.save_output(fake_out, "fake_out")
                self.save_output(disc_loss, "disc_loss")
                self.save_output(gen_loss, "gen_loss")
            print("Average prediction of real ecg: {}".format(real_avg))
            print("Average prediction of fake ecg: {}".format(fake_avg))
            print("gen loss: {}".format(gen_loss_avg))
            print("disc loss: {}".format(disc_loss_avg))
            if np.isnan(real_avg) or np.isnan(fake_avg) or any(np.isnan(disc_loss)) or any(np.isnan(gen_loss)):
                self.save_output(real_out, "real_out")
                self.save_output(fake_out, "fake_out")
                self.save_output(disc_loss, "disc_loss")
            if not os.path.exists("metrics/"):
                os.makedirs("metrics")
            f = open("metrics/output.txt", "a")
            f.write("Epoch number: {} \n".format(epoch))
            f.write("Average prediction of real images: {} \n".format(real_avg))
            f.write("Average prediction of fake images: {} \n".format(fake_avg))
            ecgs = self._generate_fake_samples(self.generator)
            epochs.append(ecgs)
            f.close()
            # save plots
            self.save_plot(
                epoch, ecgs, title=f"Generator output after {epoch} epochs and discriminator score")
            for i in range(len(ecgs)):
                ecg = ecgs[i]
                self.save_single_ecg_plot(ecg, epoch, str(
                    i), f"Generator output after {epoch} epochs")

        # save models
        self.generator.save_weights(GAN.CHECKPOINT_PATH.format("generator"))
        self.discriminator.save_weights(
            GAN.CHECKPOINT_PATH.format("discriminator"))

    @tf.function
    def _train_step(self, real_ecg):
        """
        The actual train step. For each batch the generator generates a fake batch
        then both get fed into the discriminator. Calculates the loss and gradients
        and applies them to the models.
        """
        noise = tf.random.normal([GAN.BATCH_SIZE, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_ecg = self.generator(noise, training=True)

            real_output = self.discriminator(real_ecg, training=True)
            fake_output = self.discriminator(generated_ecg, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                -1*disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return real_output, fake_output, disc_loss, gen_loss

    def save_plot(self, epoch, images, title=None):
        """
        Saves plots after each epoch under the images/ directory.
        """
        outputs = self.discriminator(images)
        x = tf.reshape(outputs, [-1])

        sort_idx = np.argsort(x)

        fig = plt.figure(figsize=(8, 8))
        if not os.path.exists(f"images/{epoch}/"):
            os.makedirs(f"images/{epoch}")
        for i in range(images.shape[0]):
            ax = plt.subplot(8, 8, i+1)
            plt.plot(images[sort_idx[i]])
            ax.axis('off')
            text = str(np.round(x[sort_idx[i]].numpy(), decimals=2))
            ax.text(0.5, -0.1, text, ha="center",
                    size=8, transform=ax.transAxes)
        if title != None:
            plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'images/{str(epoch)}/all.png')
        plt.close()

    def save_single_ecg_plot(self, ecg, epoch, name, title=None):
        """
        Saves a single ECG in a higher resolution with axes on.
        """
        # print(ecg.shape)
        # print(ecg)
        ecg_input = tf.reshape(ecg, (1, 1024))
        disc_score = self.discriminator(ecg_input)
        disc_score = np.round(disc_score.numpy(), decimals=2)
        if not os.path.exists(f"images/{epoch}/"):
            os.makedirs(f"images/{str(epoch)}")
        plt.plot(ecg)
        plt.xlabel("Samples")
        plt.ylabel("AU")
        plt.title(
            f"{title} Discriminator Score: {str(disc_score).lstrip('[').rstrip(']')}")
        plt.tight_layout()
        plt.savefig(f"images/{str(epoch)}/{name}")
        plt.close()


if __name__ == "__main__":
    """
    Defines some hyperparameters. Change here if you want to run with different settings.
    """
    gan = GAN(latent_dim=50, n_epochs=400, learning_rate_gen=1e-4,
              learning_rate_disc=1e-5, dropout_perc=0.3, max_filter=128)
    gan.train()
