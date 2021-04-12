import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import lowpy as lp



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# Prepare the training dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
batch_size    = 32
x_train       = np.reshape(x_train, (-1, 784)) / 255
x_test        = np.reshape(x_test, (-1, 784)) / 255

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
# x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# x_train = x_train[0:20]
# y_train = y_train[0:20]
# x_test = x_test[0:10]
# y_test = y_test[0:10]
# batch_size    = 16





epochs = 5
variants = 11
sigma = np.zeros(variants)
# sigma = np.logspace(-1*variants+2,0,variants-1)
# sigma = np.insert(sigma,0,0,axis=0)
# decay = np.linspace(1.0,0.94,variants)
decay = np.ones(variants)
# precision = np.power(2,np.arange(variants)+1)[::-1]
precision = np.zeros(variants)
# lower_saf = np.linspace(0,0.1,variants)
# zero_saf = np.linspace(0,0.1,variants)
# upper_saf = np.linspace(0,0.1,variants)
lower_saf = np.zeros(variants)
zero_saf = np.zeros(variants)
upper_saf = np.zeros(variants)
# RTN = np.logspace(-1*variants+2,0,variants-1)
# RTN = np.insert(RTN,0,0,axis=0)
RTN = np.zeros(variants)
# upper_drift = np.linspace(0,0.1,variants)
upper_drift = np.zeros(variants)
# lower_drift = np.linspace(0,0.1,variants)
lower_drift = np.zeros(variants)
# zero_drift = np.linspace(0,0.1,variants)
zero_drift = np.zeros(variants)
# bound_drift = np.linspace(0,0.1,variants)  
bound_drift = np.zeros(variants)
drop_destination = -0.1
drop_threshold = np.flip(np.logspace(-1*variants+2,0,variants-1)*-1)

history = lp.metrics()


for v in range(variants):
    tf.keras.backend.clear_session()
    simulator = lp.wrapper(history,
      variability_stdev=sigma[v], 
      decay=decay[v],
      precision=precision[v],
      upper_bound=0.1,
      lower_bound=-0.1,
      percent_stuck_at_lower_bound=lower_saf[v],
      percent_stuck_at_zero=zero_saf[v],
      percent_stuck_at_upper_bound=upper_saf[v],
      rtn_stdev=RTN[v],
      drift_rate_to_lower=lower_drift[v],
      drift_rate_to_upper=upper_drift[v],
      drift_rate_to_zero=zero_drift[v],
      drift_rate_to_bounds=bound_drift[v],
      drop_destination=drop_destination,
      drop_threshold=drop_threshold[v]
    )

    inputs = keras.Input(shape=(784,), name="digits")
    outputs = layers.Dense(10, name="predictions")(inputs)

    # inputs = keras.Input(shape=(784,), name="digits")
    # x1 = layers.Dense(533, activation="relu")(inputs)
    # outputs = layers.Dense(10, name="predictions")(x1)

    # inputs = keras.Input(shape=(28,28,1), name="digits")
    # c1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    # p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    # c2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(p1)
    # p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    # f1 = layers.Flatten()(p2)
    # d1 = layers.Dropout(0.5)(f1)
    # outputs = layers.Dense(10, activation="softmax")(d1)

    # inputs = tf.keras.Input(shape=(None,), dtype="int32")
    # x = tf.keras.layers.Embedding(max_features, 128)(inputs)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True),name="LowPy-bidirectional1")(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64),name="LowPy-bidirectional2")(x)
    # outputs = tf.keras.layers.Dense(2, activation="sigmoid",name="LowPy-output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer,loss_function,metrics=[keras.metrics.SparseCategoricalAccuracy()])

    simulator.post_gradient_application = [
      simulator.asymmetric_weight_drop
    ]
    

    simulator.wrap(model,optimizer,loss_function)
    simulator.plot(drop_threshold)


    

    with tf.device('/CPU:0'):
        simulator.fit(x=x_train, y=y_train, batch_size=32, epochs=2, variant=drop_threshold[v], validation_split=0.1)

    simulator.metrics.export_weights("Variant"+str(v), simulator.cell_updates)