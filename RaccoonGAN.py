import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import cv2
import time

from IPython import display

def main():
    train_path = "/home/jacob/Documents/ImageClassification/racconsVredpandas/train"
    train_raccoon_dir = os.path.join(train_path, 'raccoons')

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = image_generator.flow_from_directory(directory=train_raccoon_dir,
                                                         batch_size=32,
                                                         target_size=(56,56),
                                                         classes={"raccoons"},
                                                         color_mode="grayscale")

    generator = make_generator()

    noise = tf.random.normal([1, 100])

    gen_img = generator(noise, training=False)

    plt.imshow(gen_img[0, :, :, 0], cmap='gray')
    plt.savefig("test.png")

    critic = make_critic()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def critic_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    critic_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 50
    noise_dim = 100
    examples_to_gen = 16

    seed = tf.random.normal([examples_to_gen, noise_dim])

    @tf.function
    def train_steps(image):
        print("training a step")
        noise = tf.random.normal([128, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_imgs = generator(noise, training=True)

            fake_out = critic(gen_imgs, training=True)
            real_out = critic(image, training=True)

            gen_loss = generator_loss(fake_out)
            crit_loss = critic_loss(real_out, fake_out)

        gradis_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradis_of_crit = disc_tape.gradient(crit_loss, critic.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradis_of_gen, generator.trainable_variables))
        critic_optimizer.apply_gradients(zip(gradis_of_crit, critic.trainable_variables))


    def train(dataset, epochs):
        counter = 0
        img_counter = 0
        for epoch in range(epochs):
            print("Training Epoch ", counter)
            counter += 1
            start = time.time()

            img = next(dataset)
            while img_counter < 200:
                print("Image ", img_counter, " trained")
                img_counter += 1
                train_steps(img[0])
                img = next(dataset)
            img_counter = 0
            print()

            display.clear_output(wait=True)
            generate_and_save(generator, epoch + 1, seed)

            print("Epoch #", epoch, " Time: ", time.time() - start)

        display.clear_output(wait=True)
        generate_and_save(generator, epochs, seed)

    def generate_and_save(model, epoch, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[1, :, :, 0] * 127.5 * 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{}.png'.format(epoch))

    train(train_data_gen, EPOCHS)


def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 1)

    return model

def make_critic():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[56,56,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (5, 5), strides=(2,2), padding='same', input_shape=[56,56,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


main()
