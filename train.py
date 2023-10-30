import keras
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

epochs = 100
batch_size = 250
l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
learning_rate = 0.25

model = keras.Sequential([
    keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(100, 150, 1)),
    keras.layers.MaxPool2D(2, 1),
    keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
    keras.layers.MaxPool2D(2, 1),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)
])

optimizer = tensorflow_privacy.DPKerasSGDOptimizer(l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
                                                   num_microbatches=num_microbatches, learning_rate=learning_rate)

loss = keras.losses.BinaryCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          validation_data=(test_data, test_labels),
          batch_size=batch_size)

compute_dp_sgd_privacy.compute_dp_sgd_privacy_statement(number_of_examples=train_data.shape[0],
                                              batch_size=batch_size,
                                              noise_multiplier=noise_multiplier,
                                              num_epochs=epochs,
                                              delta=1e-5)