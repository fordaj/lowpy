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
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
batch_size    = 32
x_train       = np.reshape(x_train, (-1, 784)) / 255
x_test        = np.reshape(x_test, (-1, 784)) / 255
x_val         = x_train[-10000:]
y_val         = y_train[-10000:]
x_train       = x_train[:-10000]
y_train       = y_train[:-10000]
x_test        = tf.constant(x_test)
y_test        = tf.constant(y_test)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset   = val_dataset.batch(batch_size)

y_test = np.int64(y_test)






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
bound_drift = np.linspace(0,0.1,variants)  
# bound_drift = np.zeros(variants)


history = lp.metrics()

for v in range(variants):
    tf.keras.backend.clear_session()
    simulator = lp.wrapper(history,
      sigma=sigma[v], 
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
      drift_rate_to_bounds=bound_drift[v]
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

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer,loss_function,metrics=[keras.metrics.SparseCategoricalAccuracy()])

    simulator.pre_train_forward_propagation = [
      simulator.apply_drift
    ]

    simulator.wrap(model,optimizer,loss_function)
    simulator.plot(bound_drift)
    
    simulator.add_test_metrics_entry(0.5,0.8, bound_drift[v])

    simulator.add_test_metrics_entry(0.4,0.85, bound_drift[v])
    simulator.add_test_metrics_entry(0.3,0.9, bound_drift[v+1])

    model.fit(x_train, y_train, batch_size=32)

    print(simulator.evaluate(x_test,y_test))


    with tf.device('/GPU:0'):
        simulator.fit(epochs, train_dataset, variant_iteration=v)
    tf.keras.backend.clear_session()
    del model
    del optimizer
    del loss_function