import tensorflow as tf
from tensorflow.keras import Model, layers, losses
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, BooleanOptionalAction
import json
import os

from constants import (DATASET, MODELS_DIR, SEP, EPOCHS, MOMENTUM, MEAN, SIGMA, BATCH_SIZE,
                       TARGET_ACCURACY, MAX_TRANSLATION, MAX_ROTATION, MAX_ZOOM, STEPS)


class AdaptiveAugmenter(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.)

        # blitting and geometric augmentations are the most helpful in the low-data regime
        self.augmenter = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=train_ds_shape[1:]),
                # blitting/x-flip:
                layers.RandomFlip("horizontal"),
                # blitting/integer translation:
                layers.RandomTranslation(
                    height_factor=MAX_TRANSLATION,
                    width_factor=MAX_TRANSLATION,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                layers.RandomRotation(factor=MAX_ROTATION),
                # geometric/isotropic and anisotropic scaling:
                layers.RandomZoom(
                    height_factor=(-MAX_ZOOM, 0.0), width_factor=(-MAX_ZOOM, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images):
        augmented_images = self.augmenter(images)

        # during training either the original or the augmented images are selected
        # based on self.probability
        augmentation_values = tf.random.uniform(shape=(images.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, self.probability)
        return tf.where(augmentation_bools, augmented_images, images)

    def update(self, current_accuracy):
        # the augmentation probability is updated based on the accuracy
        accuracy_error = current_accuracy - TARGET_ACCURACY
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / STEPS, 0.0, 1.0
            )
        )


class MobileNetV2(tf.keras.Model):
    def __init__(self, in_shape, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.in_shape = in_shape
        self.augmenter = AdaptiveAugmenter()
        self.augmenter.build((batch_size, *in_shape))

    def train_step(self, data):
        imgs, labels = data

        # Augment the images before sending them to the model
        imgs = self.augmenter(imgs)
        outputs = super().train_step((imgs, labels))

        # Updating augmenter probability based on the accuracy change
        self.augmenter.update(self.get_metrics_result()['accuracy'])
        outputs['aug_p'] = self.augmenter.probability
        return outputs


def add_noise(img, label):
    "Adds gaussian noise to the image tensor."
    return img + tf.random.normal(img.shape, mean=MEAN, stddev=SIGMA), label


def sample_images(model, data, labels, class_names):
    # Sampling predicted images
    r, c = 5, 5
    items = np.random.randint(0, len(data), 25, dtype='int32')

    fig, axs = plt.subplots(r, c, figsize=(7, 7))
    tp = 0
    for row in range(r):
        for col in range(c):
            item = items[row*r+col]
            result = model((data[item]/255).reshape(1, *data[item].shape), training=False)
            if labels[item][0] == np.argmax(result):
                tp += 1
            axs[row, col].set_title(f'P"{class_names[np.argmax(result)]}" GT"{class_names[labels[item][0]]}"',
                                    fontstyle='italic', fontsize='small')
            axs[row, col].imshow(data[item])
            axs[row, col].axis('off')
    print(f'Accuracy: {tp/25}')
    fig.tight_layout(pad=3)
    plt.show()
    plt.close()


def get_MobileNetV2_model(input_shape, batch_size, num_classes, name_postfix):
    # Number of blocks differs from the original architecture because of the cifar10 image size
    inverted_residual_settings = [
        # t, c, n, s
        [1, 32, 1, 1],
        [6, 64, 3, 1],
        [6, 128, 3, 2],
        [6, 256, 1, 1],
    ]

    def expansion_block(inputs, t, out_channels, prefix):
        out_channels = t * out_channels
        out = layers.Conv2D(out_channels, 1, padding='same', use_bias=False, name=prefix + 'expand')(inputs)
        out = layers.BatchNormalization(name=prefix + 'expand_bn')(out)
        out = layers.ReLU(6, name=prefix + 'expand_relu6')(out)
        return out

    def depthwise_block(inputs, stride, prefix):
        out = layers.DepthwiseConv2D(3, stride, padding='same', use_bias=False, name=prefix + 'depthwise_conv')(inputs)
        out = layers.BatchNormalization(name=prefix + 'dw_bn')(out)
        out = layers.ReLU(6, name=prefix + 'dw_relu6')(out)
        return out

    def projection_block(inputs, out_channels, prefix):
        out = layers.Conv2D(out_channels, 1, padding='same', use_bias=False, name=prefix + 'compress')(inputs)
        out = layers.BatchNormalization(name=prefix + 'compress_bn')(out)
        return out

    def bottleneck(inputs, t, e_out_channels, p_out_channels, stride, block_id):
        out = expansion_block(inputs, t, e_out_channels, block_id)
        out = depthwise_block(out, stride, block_id)
        out = projection_block(out, p_out_channels, block_id)
        if out.shape == inputs.shape:
            out = layers.Add()([inputs, out])
        return out

    inputs = layers.Input(shape=input_shape)
    # Normalize pixel values to be between 0 and 1
    out = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='1st_conv')(inputs)
    out = layers.BatchNormalization(name='conv1_bn')(out)
    out = layers.ReLU(6, name='conv1_relu')(out)

    # Bottleneck blocks
    for b_id, setting in enumerate(inverted_residual_settings, 1):
        t, p_out_channels, num_iterations, stride = setting
        for i in range(num_iterations):
            block_id = f'{b_id}{i}'
            if block_id == '10':
                # expansion_block is skipped for the first bottleneck
                out = depthwise_block(out, stride, block_id)
                out = projection_block(out, p_out_channels, block_id)
            else:
                out = bottleneck(out, t, out.shape[-1], p_out_channels,
                                 stride if i == 1 else 1, block_id)

    out = layers.Conv2D(640, 1, padding='same', use_bias=False, name='pre_last_conv')(out)
    out = layers.BatchNormalization(name='last_bn')(out)
    out = layers.ReLU(6, name='last_relu6')(out)
    out = layers.AveragePooling2D(out.shape[1], name='average_pool')(out)
    out = layers.Conv2D(num_classes, 1, padding='same', use_bias=False, name='last_conv')(out)
    out = layers.Flatten(name='flatten')(out)
    out = layers.Softmax(name='softmax')(out)
    return MobileNetV2(input_shape, batch_size, inputs=inputs, outputs=out, name=f'MobileNetV2_{name_postfix}')


parser = ArgumentParser(prog='MobileNetV2',
                        description='Classification model for cifar10 dataset.')
parser.add_argument('-t', action=BooleanOptionalAction, help='Flag for the train mode')
parser.add_argument('-e', type=int, default=EPOCHS, help='Number of epochs to train')
parser.add_argument('-b', type=int, default=BATCH_SIZE,
                    help='Batch size. Please mind the memory limits of your system.')
args = parser.parse_args()
train_mode, epochs, batch_size = args.t, args.e, args.b

# Reading labels
with open(f'{DATASET}-labels.json', 'r') as f:
    data = json.load(f)
    class_names = data['labels']
    train_ds_shape = data['train_data_shape']
    test_ds_shape = data['test_data_shape']

print(f'Training data shape: {train_ds_shape}, test data shape: {test_ds_shape}')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Creating tf.data.Dataset objects out of the x_train and x_test to normalize and add noise to the data
if train_mode:
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .batch(batch_size, drop_remainder=True) \
            .map(lambda x, y: (layers.Rescaling(1./255.)(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
            .map(add_noise, num_parallel_calls=tf.data.AUTOTUNE) \
            .cache() \
            .shuffle(10 * batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(batch_size, drop_remainder=True) \
            .map(lambda x, y: (layers.Rescaling(1./255.)(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
            .cache() \
            .shuffle(10 * batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == "__main__":

    model = get_MobileNetV2_model(train_ds_shape[1:], batch_size, len(class_names), DATASET)

    # Save the best model based on the validation KID metric
    checkpoint_path = f'{MODELS_DIR}{SEP}{model.name}{SEP}{model.name}.ckpt'

    if train_mode:
        os.makedirs(f'{MODELS_DIR}{SEP}{model.name}', exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="auto",
            save_best_only=True,
            save_freq='epoch',
            verbose=1
            )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=MOMENTUM),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.summary()

        model.fit(train_ds,
                  epochs=epochs,
                  validation_data=test_ds,
                  callbacks=[checkpoint_callback])

    else:
        model.load_weights(checkpoint_path)
        sample_images(model, x_test, y_test, class_names)
