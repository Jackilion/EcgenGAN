"""
Defines the losses used for the cGAN.
"""

import utils
import tensorflow as tf

EPSILON = 0.0001


_discriminator_loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0.1)
_generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    # real_output = tf.sigmoid(real_output)
    # fake_output = tf.sigmoid(fake_output)

    loss1 = 1*tf.keras.backend.mean(tf.math.log(real_output + EPSILON))
    loss2 = 1 * \
        tf.keras.backend.mean(tf.math.log(tf.math.subtract(
            tf.ones_like(fake_output), fake_output) + EPSILON))
    return loss1 + loss2

    real_loss = _discriminator_loss(tf.ones_like(real_output), real_output)
    fake_loss = _generator_loss(tf.zeros_like(fake_output), fake_output)
    return - (real_loss + fake_loss)
    #! Label smoothing:
    one_minus_epsilon = tf.subtract(tf.ones_like(fake_output), tf.math.abs(
        tf.random.normal(shape=fake_output.shape, mean=0.0, stddev=0.05)))
    zero_plus_epsilon = tf.add(tf.zeros_like(fake_output), tf.math.abs(
        tf.random.normal(shape=fake_output.shape, mean=0.0, stddev=0.05)))
    loss1 = tf.keras.backend.mean(tf.math.log(real_output + EPSILON))
    loss2 = tf.keras.backend.mean(tf.math.log(tf.math.subtract(
        tf.ones_like(fake_output), fake_output) + zero_plus_epsilon))
    # if tf.math.is_nan(loss1 + loss2):
    #     print("FOUND A NAN IN LOSS!!!")
    #     print("Args: {}, {}".format(tf.math.reduce_mean(
    #         real_output), tf.math.reduce_mean(fake_output)))
    return - (loss1 + loss2)


def generator_loss(fake_output):
    return tf.keras.backend.mean(tf.math.subtract(tf.ones_like(fake_output), fake_output))

    # return _generator_loss(tf.ones_like(fake_output), fake_output)
    # fake_output = tf.sigmoid(fake_output)
    # return tf.keras.backend.mean(tf.math.subtract(tf.ones_like(fake_output), fake_output))
    return tf.keras.backend.mean(tf.math.log(tf.ones_like(fake_output) - fake_output + EPSILON))
    # WGAN:
    # return -1*tf.keras.backend.mean(fake_output)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def l1_loss(predictions, targets):
    return tf.math.abs(predictions - targets)


def exp_l1_loss(predictions, targets):
    return tf.math.exp(4*tf.math.abs(predictions - targets))


def binned_dist_loss(n, codes, output):
    codes_shifted = codes
    output_shifted = output
    losses = []

    loss_bin0 = tf.nn.sigmoid_cross_entropy_with_logits(codes, output)
    loss_bin0 = tf.reduce_mean(loss_bin0)
    losses.append(loss_bin0)

    for i in range(n):
        codes_shifted_new = tf.roll(codes_shifted, shift=-1, axis=0)
        added = tf.add(codes_shifted, codes_shifted_new)
        codes_shifted = added[:, ::2]
        codes_normalized = tf.divide(
            tf.subtract(
                codes_shifted,
                tf.reduce_min(codes_shifted)
            ),
            tf.subtract(
                tf.reduce_max(codes_shifted),
                tf.reduce_min(codes_shifted)
            )
        )

        output_shifted_new = tf.roll(output_shifted, shift=-1, axis=0)
        added = tf.add(output_shifted, output_shifted_new)
        output_shifted = added[:, ::2]
        outputs_normalized = tf.divide(
            tf.subtract(
                output_shifted,
                tf.reduce_min(output_shifted)
            ),
            tf.subtract(
                tf.reduce_max(output_shifted),
                tf.reduce_min(output_shifted)
            )
        )

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            codes_normalized, outputs_normalized)
        losses.append(1 * tf.reduce_mean(loss))
    return sum(losses) / len(losses)


def auxillary_loss(codes, aux_output):
    # diff = tf.math.subtract(codes, tf.math.sigmoid(aux_output))
    # abs_diff = tf.math.abs(diff)
    loss0 = tf.nn.sigmoid_cross_entropy_with_logits(codes, aux_output)
    # for i in range(aux_output.get_shape().as_list()[1]):
    #    aux_output[i, :] = utils.logits_to_one_hot(aux_output[i, :])
    for i in range(aux_output.get_shape().as_list()[1]):
        n_hot_row = utils.logits_to_one_hot(aux_output[i, :])
        aux_output = tf.tensor_scatter_nd_update(
            aux_output, tf.constant([[i]]), [n_hot_row])

    # loss1 = tf.nn.sigmoid_cross_entropy_with_logits(codes, aux_output)
    loss1 = binned_dist_loss(4, codes, aux_output)
    return loss0 + loss1

    # loss = tf.math.reduce_sum(abs_diff)
    # loss = tf.keras.backend.mean(diff)
    # return loss
    # return binned_dist_loss(4, codes, aux_output)
