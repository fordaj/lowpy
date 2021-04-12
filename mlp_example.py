# Import libraries
import tensorflow as tf
import numpy as np
import lowpy as lp

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
batch_size    = 32
x_train       = np.reshape(x_train, (-1, 784)) / 255 # Flatten and normalize
x_test        = np.reshape(x_test, (-1, 784)) / 255 # Flatten and normalize


# Define hyperparameters
epochs = 5
batch_size = 32
variants = 7
# Define variability_stdev = [0 1e-5 1e-4 1e-3 1e-2 1e-1 1e0]:
variability_stdev = np.logspace(-1*variants+2,0,variants-1)
variability_stdev = np.insert(variability_stdev,0,0,axis=0)


# Tracking performance metrics
history = lp.metrics()


# Variant loop
for v in range(variants): # 7 variants total
    
    # Define a model
    inputs = tf.keras.Input(shape=(784,), name="input")
    hidden = tf.keras.layers.Dense(533, activation="relu", name="lowpy-hidden")(inputs)
    outputs = tf.keras.layers.Dense(10, name="lowpy-output")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer,loss_function,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    # Configure LowPy simulator
    simulator = lp.wrapper(history, variability_stdev=variability_stdev[v])
    simulator.wrap(model, optimizer, loss_function)
    simulator.post_gradient_application = [
        simulator.write_variability
    ]
    simulator.plot(variability_stdev)

    # Fit model with GPU
    with tf.device('/GPU:0'):
        simulator.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, variant=variability_stdev[v], validation_split=0.1)