import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd
import numpy as np
from progress.bar import IncrementalBar

class wrapper:
    def __init__(self,
        metrics,
        sigma=0.0, 
        decay=1.0, 
        precision=0, 
        upper_bound=0.1, 
        lower_bound=-0.1, 
        percent_stuck_at_lower_bound=0, 
        percent_stuck_at_zero=0, 
        percent_stuck_at_upper_bound=0,
        noise=0
    ):
        self.history = metrics
        self.sigma = sigma
        self.decay = decay
        self.precision = precision
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.noise = noise
        self.range = abs(self.upper_bound) + abs(self.lower_bound)
        self.num_lower_saf = percent_stuck_at_lower_bound
        self.num_zero_saf = percent_stuck_at_zero
        self.num_upper_saf = percent_stuck_at_upper_bound
        self.tff_write_variability = tf.function(self.write_variability)
        self.tff_apply_decay = tf.function(self.apply_decay)
        self.tff_truncate_center_state = tf.function(self.truncate_center_state)
        self.tff_training_step = tf.function(self.training_step)

    def wrap(self,model,optimizer,loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def plot(self,varied_parameter):
        self.header = varied_parameter

    def weight_zeros(self):
        weights = self.model.trainable_weights
        self.zeros = []
        for w in weights:
            self.zeros.append(tf.Variable(tf.zeros(w.shape,dtype=tf.dtypes.float32)))

    def initialization_variability(self):
        weights = []
        for l in self.model.layers:
            for w in range(len(l.weights)):
                if (not 'conv' in l.weights[w].name) and (not 'embed' in l.weights[w].name):
                    weights.append(tf.random.normal(l.weights[w].shape,mean=l.weights[w],stddev=self.sigma))

    def initialize_stuck_at_fault_matrices(self):
        self.stuck_at_lower_bound_matrix = []
        self.stuck_at_zero_matrix = []
        self.stuck_at_upper_bound_matrix = []
        for l in self.model.layers:
            for w in range(len(l.weights)):
                weight_dims = l.weights[w].shape
                if (not 'conv' in l.weights[w].name) and (not 'embed' in l.weights[w].name):
                    num_weights = tf.reduce_prod(weight_dims).numpy()
                    num_lower = round(self.num_lower_saf * num_weights)
                    num_zero = round(self.num_zero_saf * num_weights)
                    num_upper = round(self.num_upper_saf * num_weights)
                    stuck = np.zeros(num_weights)
                    lower = np.zeros(num_weights)
                    zero = np.zeros(num_weights)
                    upper = np.zeros(num_weights)
                    stuck[0:num_lower] = 1
                    stuck[num_lower:(num_lower+num_zero)] = 2
                    stuck[(num_lower+num_zero):(num_lower+num_zero+num_upper)] = 3
                    np.random.shuffle(stuck)
                    for s in range(len(stuck)):
                        if (stuck[s] == 1):
                            lower[s] = 1
                        elif (stuck[s] == 2):
                            zero[s] = 1
                        elif (stuck[s] == 3):
                            upper[s] = 1
                    lower = tf.reshape(lower,weight_dims)
                    zero = tf.reshape(zero,weight_dims)
                    upper = tf.reshape(upper,weight_dims)
                else:
                    lower = tf.zeros(weight_dims,dtype=l.weights[w].dtype)
                    zero = tf.zeros(weight_dims,dtype=l.weights[w].dtype)
                    upper = tf.zeros(weight_dims,dtype=l.weights[w].dtype)
                self.stuck_at_lower_bound_matrix.append(lower)
                self.stuck_at_zero_matrix.append(zero)
                self.stuck_at_upper_bound_matrix.append(upper)

    @tf.function
    def write_variability(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                weights[w].assign(tf.random.normal(weights[w].shape,mean=weights[w],stddev=self.sigma))
        self.optimizer.apply_gradients(zip(self.zeros,weights))

    @tf.function
    def apply_decay(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                weights[w].assign(tf.multiply(weights[w],self.decay))
        self.optimizer.apply_gradients(zip(self.zeros,weights))

    @tf.function
    def truncate_center_state(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                one = tf.add(weights[w],abs(self.lower_bound))
                two = tf.multiply(one,self.precision/self.range)
                three = tf.clip_by_value(two,clip_value_min=0,clip_value_max=self.precision)
                four = tf.round(three)
                five = tf.divide(four,self.precision/self.range)
                six = tf.subtract(five,abs(self.lower_bound))
                weights[w].assign(six)
        self.optimizer.apply_gradients(zip(self.zeros,weights))
    
    @tf.function
    def apply_stuck_at_faults(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                not_stuck_at_lower_bound = tf.cast(tf.math.round((self.stuck_at_lower_bound_matrix[w] - 1) * -1),weights[w].dtype)
                not_stuck_at_zero        = tf.cast(tf.math.round((self.stuck_at_zero_matrix[w] - 1) * -1),weights[w].dtype)
                not_stuck_at_upper_bound = tf.cast(tf.math.round((self.stuck_at_upper_bound_matrix[w] - 1) * -1),weights[w].dtype)
                bounds = tf.cast((self.lower_bound*self.stuck_at_lower_bound_matrix[w]) + (self.upper_bound*self.stuck_at_upper_bound_matrix[w]),weights[w].dtype)
                weights[w].assign(bounds + weights[w] * not_stuck_at_lower_bound * not_stuck_at_zero * not_stuck_at_upper_bound)
        self.optimizer.apply_gradients(zip(self.zeros,weights))

    def training_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            logits = self.model(x_batch_train, training=True)
            loss_value = self.loss_function(y_batch_train, logits)
        return tape.gradient(loss_value, self.model.trainable_weights)
    
    def apply_grads(self):
        self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))

    def evaluate(self):
        logits = self.model(self.x_test)
        loss = self.loss_function(self.y_test, logits)
        one = tf.argmax(logits,1)
        two = tf.cast(self.y_test,one.dtype)

        accuracy = tf.math.count_nonzero(tf.math.equal(one,two)) / len(self.y_test)
        return [loss.numpy(),accuracy.numpy()]

    def fit(self, x_test, y_test, epochs, train_dataset,variant_iteration=0):
        self.x_test = x_test
        self.y_test = y_test
        self.weight_zeros()
        self.initialization_variability()
        self.initialize_stuck_at_fault_matrices()
        test_loss = []
        test_accuracy = []
        test_metrics = self.evaluate()
        test_loss.append(test_metrics[0])
        test_accuracy.append(test_metrics[1])
        print("--------------------------")
        print("Baseline\tLoss: ", "{:.4f}".format(test_metrics[0]), "\tAccuracy: ", "{:.2f}".format(test_metrics[1]*100),"%")
        for epoch in range(epochs):
            with IncrementalBar('Epoch ' + str(epoch), max=len(train_dataset), suffix="%(index)d/%(max)d - %(eta)ds\tLoss: " + "{:.4f}".format(test_metrics[0]) + "\tAccuracy: " + "{:.2f}".format(test_metrics[1]*100) + "%%") as bar:
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.apply_stuck_at_faults()
                    self.grads = self.training_step(tf.constant(x_batch_train), tf.constant(y_batch_train))
                    self.apply_grads()
                    if self.precision > 0:
                        self.truncate_center_state()
                    self.write_variability()
                    self.apply_decay()
                    bar.next()
                test_metrics =  self.evaluate()
                test_loss.append(test_metrics[0])
                test_accuracy.append(test_metrics[1])
        print("\tFinal loss: ", test_metrics[0], "\tAccuracy: ", test_metrics[1]*100,"%",end='')
        self.history.test.loss[self.header[variant_iteration]] = test_loss
        self.history.test.accuracy[self.header[variant_iteration]] = test_accuracy
        self.history.test.loss.to_csv(self.history.testDir + "/Loss.csv")
        self.history.test.accuracy.to_csv(self.history.testDir + "/Accuracy.csv")
        tf.keras.backend.clear_session()
        del self.model
        del self.optimizer
        del self.loss_function

