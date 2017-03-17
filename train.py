# ===================================================================
# Implementation of batch normalization on a simple feedforward
# network with 2 hidden layers.
#
#       Batch Normalization: Accelerating Deep Network Training by
#       Reducing Internal Covariate Shift
#
#       by Sergey Ioffe and Christian Szegedy
#
#       https://arxiv.org/pdf/1502.03167.pdf
#
# NOTE: Most people have been using an exponential moving average
# for the estimation of the statistics of the population
# distribution, instead of a simple moving average.
# It's also definitely easier to implement so I went with that.
#
# Many thanks to http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
# for some much needed clarification with some tensorflow operations.
#
# written by Dimitris Kalatzis
# ===================================================================

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def batch_norm_transform(inputs, epsilon, training, ema_decay):
    """
    Batch normalization of activations.

    :param inputs: NumPy array/TensorFlow tensor object - the activations of the
    previous layer (inputs for the first hidden layer)
    :param epsilon: float - a small float constant for numerical stability
    :param training: boolean - indicates training or inference phase
    :param ema_decay: float - decay parameter for the exponential moving average op
    :return: TensorFlow tensor objects representing the
    shift, scale and batch-normalized activations
    """
    batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])

    # scale and shift params
    gamma = tf.get_variable("scale", initializer=tf.ones([inputs.get_shape()[1]]))
    beta = tf.get_variable("shift", initializer=tf.zeros([inputs.get_shape()[1]]))

    # tensors for the population mean and variance
    pop_mean = tf.get_variable("pop_mean", initializer=tf.zeros(inputs.get_shape()[1]),
                               trainable=False)
    pop_var = tf.get_variable("pop_var", initializer=tf.zeros(inputs.get_shape()[1]),
                              trainable=False)

    if training:
        # during training produce estimates of the population
        # mean and variance for use during inference
        true_mean = tf.assign(pop_mean, pop_mean * ema_decay + batch_mean * (1 - ema_decay))
        true_var = tf.assign(pop_var, pop_var * ema_decay + batch_var * (1 - ema_decay))

        with tf.control_dependencies([true_mean, true_var]):
            in_norm = tf.divide(tf.subtract(inputs, batch_mean), tf.sqrt(batch_var + epsilon))
    else:
        # during inference the estimates for the population mean
        # and variance are used
        in_norm = tf.divide(tf.subtract(inputs, pop_mean), tf.sqrt(pop_var + epsilon))

    # scale and shift the normalized value
    batch_norm = tf.add(tf.multiply(gamma, in_norm), beta)

    return gamma, beta, batch_norm


def feedforward_BN(inputs, n_in, n_classes, hidden, epsilon, ema_decay, training=True):
    """
    Function for building the neural network graph

    :param inputs: NumPy array
    :param n_in: int - input dimensions
    :param n_classes: int - class dimensions
    :param hidden: int - hidden layer dimensions
    :param epsilon: float - a small float constant for numerical stability
    :param training: boolean - indicates training or inference phase
    :return: TensorFlow tensor object representing the network output
    """
    with tf.variable_scope("layer1"):
        W1 = tf.get_variable("weights", initializer=tf.random_normal([n_in, hidden]))
        gamma1, beta1, y1_BN = batch_norm_transform(inputs, epsilon, training, ema_decay)
        layer1_BN = tf.matmul(y1_BN, W1)

    with tf.variable_scope("layer2"):
        W2 = tf.get_variable("weights", initializer=tf.random_normal([hidden, hidden]))
        gamma2, beta2, y2_BN = batch_norm_transform(layer1_BN, epsilon, training, ema_decay)
        layer2_BN = tf.nn.relu(tf.matmul(y2_BN, W2))

    with tf.variable_scope("out_layer"):
        W_out = tf.get_variable("weights", initializer=tf.random_normal([hidden, n_classes]))
        b_out = tf.get_variable("biases", initializer=tf.random_normal([n_classes]))
        layer_out = tf.add(tf.matmul(layer2_BN, W_out), b_out)

    return layer_out


def train(dataset, batch_size, l_rate, epochs, validation_freq, patience):
    """
    Main training function. Modified implementation of early stopping.
    Final model is saved.

    :param dataset: NumPy array
    :param batch_size: int - size of mini-batches
    :param l_rate: float - learning rate of SGD optimization
    :param epochs: int - training epochs
    :param validation_freq: int - the frequency of checking
    model performance on validation set
    :param patience: int - wait for this many epochs for
    model performance to improve before stopping training
    :return: None
    """
    features = 784
    classes = 10

    x = tf.placeholder(tf.float32, [None, features], "Inputs")
    y = tf.placeholder(tf.int32, [None, ], "Labels")
    y_one_hot = tf.one_hot(y, classes)

    predictions = feedforward_BN(x, features, classes, 128, 1e-5, 0.99)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_one_hot))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(loss)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        patience_timer = 0
        best_valid_acc = -np.inf

        for epoch in range(epochs):
            avg_loss = 0.

            train_batch = dataset.train.next_batch(batch_size)
            train_x = train_batch[0]
            train_y = train_batch[1]

            _, c = sess.run([optimizer, loss],
                            feed_dict={x: train_x,
                                       y: train_y})

            avg_loss += c / batch_size

            print("Epoch:", epoch, "Loss:", avg_loss)

            if epoch % validation_freq == 0:
                valid_batch = dataset.validation.next_batch(batch_size)
                valid_x = valid_batch[0]
                valid_y = valid_batch[1]

                valid_acc = sess.run(accuracy,
                                     feed_dict={x: valid_x,
                                                y: valid_y})
                print("Validation accuracy:", valid_acc)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    patience_timer = 0
                else:
                    patience_timer += 1
                    print("Patience timer:", patience_timer)
                    print("Current best validation accuracy:", best_valid_acc)
                    print("----------------")

                if patience_timer == patience:
                    print("Training complete.")
                    print("Overall Best validation accuracy:", best_valid_acc)
                    print("----------------")
                    saver.save(sess, "./saved_models/saved-model")
                    break


def test(dataset):
    """
    Main testing function. Loads the model and runs it on the test set.

    :param dataset: NumPy array
    :return: None
    """
    tf.reset_default_graph()

    features = 784
    classes = 10

    x = tf.placeholder(tf.float32, [None, features], "Test_inputs")
    y = tf.placeholder(tf.int32, [None, ], "Test_labels")
    y_one_hot = tf.one_hot(y, classes)

    predictions = feedforward_BN(x, features, classes, 128, 1e-5, 0.99, False)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, './saved_models/saved-model')

        test_x = dataset.test.images
        test_y = dataset.test.labels

        accuracy = sess.run(accuracy, feed_dict={x: test_x,
                                                 y: test_y})

        print("Test accuracy:", accuracy)


if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 100
    TEST_BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    EPOCHS = 4000
    VALID_FREQ = 5
    PATIENCE = 15
    DATASET = input_data.read_data_sets('MNIST_data')

    train(DATASET, TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, VALID_FREQ, PATIENCE)
    test(DATASET)