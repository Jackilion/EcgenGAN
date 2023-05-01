import ecg_reader
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from numpy.random import randint
import os
import math
import sys


class InfoGAN:
    BUFFER_SIZE = 70000  # The Buffer size for shuffling the dataset. MNIST has 60000 pictures
    BATCH_SIZE = 512  # How many images are contained in each batch. Higher numbers = faster, but more VRAM needed
    EVAL_BATCH_SIZE = 1024
    N_SAMPLES = 64 # How many samples to produce after each epoch for the plots.
    CHECKPOINT_PATH = "models/{}.ckpt"
    TINY = 1e-8  # used in the losses to avoid log(0)

    def __init__(self, latent_dim, n_epochs, learning_rate_gen, learning_rate_disc, dropout_perc, max_filter):
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.dropout_perc = dropout_perc
        self.max_filter = max_filter
        self.n_cat = 3  # number of categorical info inputs

        # Turn on 16 bit floating point precision which helps with training speeds
        # mixed means the computations are fp16, and the variables are stored in fp32
        # mixed_precision.set_global_policy('mixed_float16')

        #self.train_data = ecg_reader.load_data("../data/incartdb", 0, 75, 1)
        data = np.genfromtxt("../data/synthetic/data.csv", delimiter=",")
        self.train_data = np.transpose(data)
        print("length: " + str(len(self.train_data[0])))
        self.series_length = len(self.train_data[0])
        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_gen)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_disc)
        self.auxillary_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_disc)
        self.seeds = tf.random.normal(
            [int(InfoGAN.N_SAMPLES / 2), self.latent_dim])
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            self.train_data).shuffle(InfoGAN.BUFFER_SIZE).batch(InfoGAN.BATCH_SIZE)

        self._make_generator_model()
        self._make_discriminator_and_auxillary_model()

    def _make_generator_model(self):
        """ 
        This function defines the generator using TFs sequential API.
        Input are the category concatted to the latent space vector
        It upsamples this vector through a series
        of transpose convolutions until it arrives at the desired
        output dimension of (1024, 1).      
        """
        self.generator = tf.keras.Sequential()
        self.generator.add(layers.Dense(int(self.series_length / 8) * self.max_filter,
                           use_bias=False, input_shape=(self.latent_dim + self.n_cat,), batch_size=InfoGAN.BATCH_SIZE))
        self.generator.add(layers.LeakyReLU())

        self.generator.add(layers.Reshape(
            (int(self.series_length / 8), self.max_filter)))
        # assert self.generator.output_shape == (
        #    None, self.series_length / 8, self.max_filter)

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(self.max_filter, int(
            self.series_length/6), strides=1, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        # assert self.generator.output_shape == (
        #     None, int(self.series_length / 8), self.max_filter)

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(self.max_filter / 2),
                           int(self.series_length/6), strides=2, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        # assert self.generator.output_shape == (
        #     None, int(self.series_length / 4), int(self.max_filter / 2))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(self.max_filter/4), int(
            self.series_length / 40), strides=2, padding="same", use_bias=False))
        self.generator.add(layers.LeakyReLU())
        # assert self.generator.output_shape == (
        #     None, int(self.series_length / 2), int(self.max_filter / 4))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(int(
            self.max_filter/8), int(self.series_length/6), strides=2, padding="same", use_bias=True))
        self.generator.add(layers.LeakyReLU())
        # assert self.generator.output_shape == (
        #     None, self.series_length, int(self.max_filter / 8))

        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.Conv1DTranspose(
            1, int(self.series_length / 20), strides=1, padding="same", use_bias=True))
        # self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.summary()

        # assert self.generator.output_shape == (None, self.series_length, 1)
        # return self.generator

    def _make_discriminator_and_auxillary_model(self):
        """
        This function defines the discriminator using TFs functional API.
        It takes the ECG of shape (1024, 1) and applies a series of convolutional
        and pooling layers. At the end, the output gets flattened. On top of that flattened
        layer, two output heads are defined. One that predicts the category through a softmaxed
        dense layer containing 3 units, and one that predicts whether the image is real or fake.
        """
        image_input = layers.Input(
            shape=(self.series_length, 1), batch_size=InfoGAN.BATCH_SIZE)
        print("----Image input shape:----")
        print(image_input.shape)

        conv1 = layers.Conv1D(self.max_filter, int(
            self.series_length/128), strides=2, padding="same")(image_input)
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

        print(lkyReLU4.shape)


        fltn = layers.Flatten()(lkyReLU4)

        # auxillary
        out_dense = layers.Dense(128, use_bias=True)(fltn)
        out_act = layers.LeakyReLU()(out_dense)
        q_output = layers.Dense(3, activation='softmax')(out_act)
        self.auxillary = models.Model(inputs=image_input, outputs=q_output)
        print("aux model summary:")
        self.auxillary.summary()

        # discriminator
        d_output = layers.Dense(1, activation="sigmoid")(fltn)
        self.discriminator = models.Model(inputs=image_input, outputs=d_output)

        return self.discriminator, self.auxillary

    def _discriminator_loss(self, real_output, fake_output):
        """
        The loss function for the discriminator.
        """
        loss1 = 1*tf.keras.backend.mean(tf.math.log(real_output + 0.000001))
        loss2 = 1 * \
            tf.keras.backend.mean(tf.math.log(tf.math.subtract(
                tf.ones_like(fake_output), fake_output) + 0.000001))
        return loss1 + loss2

    def _generator_loss(self, fake_output):
        """
        The loss function for the generator
        """
        return tf.keras.backend.mean(tf.math.subtract(tf.ones_like(fake_output), fake_output))
        # WGAN:
        # return -1*tf.keras.backend.mean(fake_output)

    def _auxillary_loss(self, codes, aux_output):
        """
        The loss function for the auxillary head of the discriminator
        """
        loss = tf.keras.losses.CategoricalCrossentropy()
        return loss(codes, aux_output)

    def _generate_fake_samples(self, g_model, codes=None):
        """
        Calls the generator model to generate ECGs using the provided category codes. 
        Used for plotting the progress at the end of each epoch
        """
        noise = tf.random.normal([int(InfoGAN.N_SAMPLES), self.latent_dim])
        if codes == None:
            codes = tf.random.uniform(
                [InfoGAN.N_SAMPLES], minval=0, maxval=3, dtype=tf.int32)
            codes = tf.one_hot(codes, depth=3)
        #x_input = tf.concat([self.seeds, noise], 0)
        input = self._concat_inputs([noise, codes])
        X = g_model(input)
        return X, codes

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
            for batch in self.train_dataset:
                ro, fo, dl, gl = self._train_step(batch, epoch)
                real_out.append(ro)
                fake_out.append(fo)
                disc_loss.append(dl)
                gen_loss.append(gl)

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
            ecgs, codes = self._generate_fake_samples(self.generator)
            epochs.append(ecgs)
            f.close()
            # save plots
            self.save_plot(
                epoch, ecgs, codes, title=f"Generator output after {epoch} epochs, codes, and discriminator score", name="all")
            index_1 = [0 for i in range(64)]
            index_1_tf = tf.one_hot(index_1, depth=3)
            index_2 = [1 for i in range(64)]
            index_2_tf = tf.one_hot(index_2, depth=3)
            index_3 = [2 for i in range(64)]
            index_3_tf = tf.one_hot(index_3, depth=3)

            for i in range(len(ecgs)):
                ecg = ecgs[i]
                self.save_single_ecg_plot(ecg, epoch, str(
                    i), codes=codes[i], title=f"Generator output after {epoch} epochs.")

            ecg_1, codes = self._generate_fake_samples(
                self.generator, codes=index_1_tf)
            ecg_2, codes = self._generate_fake_samples(
                self.generator, codes=index_2_tf)
            ecg_3, codes = self._generate_fake_samples(
                self.generator, codes=index_3_tf)
            self.save_plot(epoch, ecg_1, index_1_tf, title=f"Generator output after {epoch} epochs for code 1 0 0 and discriminator score",
                           name="cat_1")
            self.save_plot(epoch, ecg_2, index_2_tf, title=f"Generator output after {epoch} epochs for code 0 1 0 and discriminator score",
                           name="cat_2")
            self.save_plot(epoch, ecg_3, index_3_tf, title=f"Generator output after {epoch} epochs for code 0 0 1 and discriminator score",
                           name="cat_3")

        # save models
        self.generator.save_weights(
            InfoGAN.CHECKPOINT_PATH.format("generator"))
        self.discriminator.save_weights(
            InfoGAN.CHECKPOINT_PATH.format("discriminator"))

    def _concat_inputs(self, input):
        concat_input = layers.Concatenate()(input)
        return concat_input

    @tf.function
    def _train_step(self, real_ecg, epoch):
        """
        The actual train step. For each batch the generator generates a fake batch
        then both get fed into the discriminator. Calculates the loss and gradients
        and applies them to the models.
        """
        if real_ecg.shape == (96, 1024, 1): # This is here so the function doesn't crash at the end of an epoch.
            return
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as aux_tape:
            noise = tf.random.normal([InfoGAN.BATCH_SIZE, self.latent_dim])
            codes = tf.random.categorical(tf.math.log(
                [[0.07, 0.58, 0.35]]), InfoGAN.BATCH_SIZE) 
            #Samples codes with the same frequency as they occur 
            # in the training dataset
            codes = tf.transpose(codes)

            codes = tf.one_hot(codes, depth=3)
            codes = tf.squeeze(codes)
            input = self._concat_inputs([noise, codes])
            generated_ecg = self.generator(input, training=True)

            real_output = self.discriminator(real_ecg, training=True)
            fake_output = self.discriminator(generated_ecg, training=True)
            aux_output = self.auxillary(generated_ecg, training=True)


            self.discriminator.trainable = True
            disc_loss = self._discriminator_loss(real_output, fake_output)
            gradients_of_discriminator = disc_tape.gradient(
                -1*disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            self.discriminator.trainable = False
            #freezing the core discriminator so it doesn't train twice per epoch.

            # Turn on auxillary training after the generator had some time to converge.
            gen_loss = self._generator_loss(fake_output)
            aux_loss = self._auxillary_loss(codes, aux_output)
            gradients_of_auxillary = aux_tape.gradient(
                aux_loss, self.auxillary.trainable_variables)
            if epoch > 20 or epoch == 0:
                self.auxillary_optimizer.apply_gradients(
                    zip(gradients_of_auxillary, self.auxillary.trainable_variables))
            else:
                aux_loss = tf.zeros_like(gen_loss)

            gradients_of_generator = gen_tape.gradient(
                gen_loss + aux_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))

            return real_output, fake_output, disc_loss, gen_loss

    def save_plot(self, epoch, images, codes, title=None, name=None):
        """
        Saves plots after each epoch under the images/ directory.
        """
        if not os.path.exists(f"images/{epoch}/"):
            os.makedirs(f"images/{epoch}")
        codes_npy = [code.numpy() for code in codes]
        outputs = self.discriminator(images)
        x = tf.reshape(outputs, [-1])

        sort_idx = np.argsort(x)

        fig = plt.figure(figsize=(8, 8))
        if not os.path.exists(f"images/{epoch}"):
            os.makedirs("images")
        for i in range(images.shape[0]):
            ax = plt.subplot(8, 8, i+1)
            plt.plot(images[sort_idx[i]])
            ax.axis('off')
            text = str(np.round(x[sort_idx[i]].numpy(), decimals=2))
            ax.text(0.5, -0.1, text, ha="center",
                    size=8, transform=ax.transAxes)
            ax.text(0.5, -0.3, str(codes_npy[i]).lstrip("[").rstrip("]").replace(".", ""), ha="center",
                    size=8, transform=ax.transAxes)

        if name == None:
            name = 'image_at_epoch_{:04d}.png'.format(epoch)
        if title != None:
            plt.suptitle(title)

        plt.tight_layout()
        plt.savefig(f'images/{epoch}/{name}')
        plt.close()

    def save_single_ecg_plot(self, ecg, epoch, name, codes=None, title=None):
        """
        Saves a single ECG in a higher resolution with axes on.
        """
        ecg_input = tf.reshape(ecg, (1, 1024))
        disc_score = self.discriminator(ecg_input)
        disc_score = np.round(disc_score.numpy(), decimals=2)
        if not os.path.exists(f"images/{epoch}/"):
            os.makedirs(f"images/{str(epoch)}")
        plt.plot(ecg)
        plt.xlabel("Samples")
        plt.ylabel("AU")
        plt.title(
            f"{title} Discriminator Score: {str(disc_score).lstrip('[').rstrip(']')}. Codes: {str(codes.numpy()).lstrip('[').rstrip(']').replace('.', '')}", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"images/{epoch}/{name}")
        plt.close()


if __name__ == "__main__":
    """
    Defines some hyperparameters. Change here if you want to run with different settings.
    """
    gan = InfoGAN(latent_dim=50, n_epochs=400, learning_rate_gen=1e-4,
                  learning_rate_disc=1e-5, dropout_perc=0.3, max_filter=128)
    gan.train()
