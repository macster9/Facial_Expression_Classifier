import tensorflow as tf
from tensorflow.python.keras import layers, activations
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from skimage import filters
from network import build


class CNN:
    def __init__(self, learning_rate):
        super(CNN, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 2))
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 64))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 64))
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 128))
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 128))
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 256))
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 256))
        self.conv3_4 = layers.Conv2D(256, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 256))
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 256))
        self.conv4_2 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.conv4_3 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.conv4_4 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.pool4 = layers.MaxPooling2D((2, 2))
        self.conv5_1 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.conv5_2 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.conv5_3 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.conv5_4 = layers.Conv2D(512, (3, 3), activation=activations.gelu, input_shape=(1, 48, 48, 512))
        self.flat = layers.Flatten()

        self.hidden_layer_1 = layers.Dense(4096, activation=activations.gelu)
        self.hidden_layer_2 = layers.Dense(1000, activation=activations.gelu)
        self.output = layers.Dense(7, activation=activations.sigmoid)

        self.train_vars = []
        self.collect_trainable_variables()

    def predict(self, im):
        out = self.edge_detection(im)
        out = self.conv1_1(out)
        out = self.conv1_2(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.conv3_3(out)
        out = self.conv3_4(out)
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.conv4_4(out)
        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.conv5_4(out)
        out = self.flat(out)
        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)
        out = self.output(out)
        return out

    @staticmethod
    def edge_detection(x):
        edges = np.float32(filters.prewitt(x)) ** 2
        return np.moveaxis(np.asarray([x[:, :, :, 0], edges[:, :, :, 0]]), [1, 2, 3, 0], [0, 1, 2, 3])

    def train(self, data, labels, batch_size):
        data, labels = build.shuffle(data, labels)
        data = self.batchify(data, batch_size)
        labels = self.batchify(labels, batch_size)
        loss_data = []
        with tf.GradientTape() as tape:
            for index, (data_batch, labels_batch) in enumerate(zip(data, labels)):
                for image, label in tqdm(zip(data_batch, labels_batch), desc=f"Batch {index}/{len(data)}"):
                    image, label = np.asarray([image]), np.asarray([label])
                    prediction = self.predict(image)
                    loss_data.append(self.loss(prediction, label))
                total_loss = tf.reduce_sum(loss_data)
                print(total_loss)
                grads = tape.gradient(total_loss, self.train_vars)
                self.optimizer.apply_gradients(zip(grads, self.train_vars))
        return

    @staticmethod
    def batchify(data, batch_size):
        return data.reshape(
            np.asarray((np.append([data.shape[0] / batch_size, batch_size], [data.shape[1:]])), dtype=int))

    def train_on_batch(self, x, y):
        assert x.shape[0] == y.shape[0], f"Features and labels do not match." \
                                f" Features in batches of {x.shape[0]}." \
                                f" Labels in batches of {y.shape[0]}"
        batch_size = x.shape[0]
        y_hat = tf.experimental.numpy.empty(shape=7, dtype=np.float32)
        with tf.GradientTape() as tape:
            for i in trange(batch_size, desc="Learning..."):
                image = tf.expand_dims(x[i], axis=0)
                prediction = self.predict(image)
                # if prediction[0] != 0:
                #     prediction = tf.math.log(tf.math.abs(prediction))
                y_hat = tf.concat((y_hat, prediction[0]), axis=0)
            y_hat = tf.reshape(tensor=y_hat[7:], shape=(batch_size, 7))
            print(np.amax(y_hat, axis=1).mean(), np.amin(y_hat, axis=1).mean(), np.mean(y_hat, axis=1).mean())
            print(y)
            loss = self.loss(y, y_hat)
            grads = tape.gradient(loss, self.train_vars)
            self.optimizer.apply_gradients(zip(grads, self.train_vars))
        return loss

    def test_on_batch(self, x):
        batch_size = x.shape[0]
        y_hat = tf.experimental.numpy.empty(shape=7, dtype=np.float32)
        for i in range(batch_size):
            image = tf.expand_dims(x[i], axis=0)
            prediction = self.predict(image)
            # if y_hat[0] != 0:
            #     y_hat = tf.math.log(tf.math.abs(y_hat))
            # y_hat = int(np.asarray(self.predict(image)).flatten().item())
            # if y_hat > 6.0:
            #     y_hat = 6.0
            # if (y_hat < 0) or (y_hat == "nan"):
            #     y_hat = 0.0
            # predictions.append(y_hat)
            y_hat = tf.concat((y_hat, prediction[0]), axis=0)
        y_hat = tf.reshape(tensor=y_hat[7:], shape=(batch_size, 7))
        # print(y_hat)
        return y_hat

    @staticmethod
    def loss(y_hat, y_true):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_hat)
        # return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_hat)

    def collect_trainable_variables(self):
        self.predict(np.zeros((1, 48, 48, 1), dtype=np.float32))
        self.train_vars.extend(self.conv1_1.trainable_weights)
        self.train_vars.extend(self.conv1_2.trainable_weights)
        self.train_vars.extend(self.conv2_1.trainable_weights)
        self.train_vars.extend(self.conv2_2.trainable_weights)
        self.train_vars.extend(self.conv3_1.trainable_weights)
        self.train_vars.extend(self.conv3_2.trainable_weights)
        self.train_vars.extend(self.conv3_3.trainable_weights)
        self.train_vars.extend(self.conv3_4.trainable_weights)
        self.train_vars.extend(self.conv4_1.trainable_weights)
        self.train_vars.extend(self.conv4_2.trainable_weights)
        self.train_vars.extend(self.conv4_3.trainable_weights)
        self.train_vars.extend(self.conv4_4.trainable_weights)
        self.train_vars.extend(self.conv5_1.trainable_weights)
        self.train_vars.extend(self.conv5_2.trainable_weights)
        self.train_vars.extend(self.conv5_3.trainable_weights)
        self.train_vars.extend(self.conv5_4.trainable_weights)
        self.train_vars.extend(self.hidden_layer_1.trainable_weights)
        self.train_vars.extend(self.hidden_layer_2.trainable_weights)
        self.train_vars.extend(self.output.trainable_weights)

