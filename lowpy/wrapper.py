import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from progress.bar import IncrementalBar
import pandas as pd

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
        rtn_stdev=0,
        drift_rate_to_upper=1,
        drift_rate_to_zero=1,
        drift_rate_to_lower=1,
        drift_rate_to_bounds=1
    ):
        self.history = metrics
        self.sigma = sigma
        self.rtn_stdev = rtn_stdev
        self.decay = decay
        self.precision = precision
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.range = abs(self.upper_bound) + abs(self.lower_bound)
        self.num_lower_saf = percent_stuck_at_lower_bound
        self.num_zero_saf = percent_stuck_at_zero
        self.num_upper_saf = percent_stuck_at_upper_bound
        self.drift_rate_to_upper = drift_rate_to_upper
        self.drift_rate_to_zero = drift_rate_to_zero
        self.drift_rate_to_lower = drift_rate_to_lower
        self.drift_rate_to_bounds = drift_rate_to_bounds
        self.tff_write_variability = tf.function(self.write_variability)
        self.tff_apply_decay = tf.function(self.apply_decay)
        self.tff_truncate_center_state = tf.function(self.truncate_center_state)
        self.tff_training_step = tf.function(self.step)

        self.post_initialization = []
        self.pre_train_forward_propagation = []
        self.post_train_forward_propagation = []
        self.pre_gradient_calculation = []
        self.post_gradient_calculation = []
        self.pre_gradient_application = []
        self.post_gradient_application = []
        self.pre_evaluation = []
        self.post_evaluation = []
        self.pre_inference = []
        self.post_inference = []

    def wrap(self,model,optimizer,loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def plot(self,varied_parameter):
        self.header = varied_parameter

    def weight_zeros(self):
        """
        Creates tensors of zeros identical dimensionally to that of the model's trainable weights.
        """
        weights = self.model.trainable_weights
        self.zeros = []
        for w in weights:
            self.zeros.append(tf.Variable(tf.zeros(w.shape,dtype=tf.dtypes.float32)))

    def initialization_variability(self):
        """
        Applies write variability to the weights post-initialization.
        """
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
    def apply_rtn(self):
        self.rtn_weights = self.model.trainable_weights
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                weights[w].assign(tf.random.normal(weights[w].shape,mean=weights[w],stddev=self.rtn_stdev))
        self.optimizer.apply_gradients(zip(self.zeros,weights))
    
    @tf.function
    def remove_rtn(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                weights[w].assign(self.rtn_weights[w])
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
    
    @tf.function
    def apply_drift(self):
        weights = self.model.trainable_weights
        for w in range(len(weights)):
            if (not 'conv' in weights[w].name) and (not 'embed' in weights[w].name):
                weights[w].assign((weights[w] + self.range)*(1+self.drift_rate_to_upper)-self.range)
                weights[w].assign((weights[w] + self.range)*(1-self.drift_rate_to_lower)-self.range)
                weights[w].assign(weights[w]*(1-self.drift_rate_to_zero))
                weights[w].assign(weights[w] + tf.sign(weights[w])*((self.upper_bound-tf.abs(weights[w]))*self.drift_rate_to_bounds))
        self.optimizer.apply_gradients(zip(self.zeros,weights))

    def step(self, x_batch_train, y_batch_train):

        # Pre Training Forward Propagation
        for function in self.pre_train_forward_propagation:
            function()
        # Forward Propagation
        with tf.GradientTape() as tape:
            logits = self.model(x_batch_train, training=True)
            loss_value = self.loss_function(y_batch_train, logits)
        # Post Training Forward Propagation
        for function in self.post_train_forward_propagation:
            function()

        # Pre Gradient Calculation
        for function in self.pre_gradient_calculation:
            function()
        # Gradient Calculation
        grad = tape.gradient(loss_value, self.model.trainable_weights)
        # Post Gradient Calculation
        for function in self.post_gradient_calculation:
            function()

        return grad
    
    def apply_gradients(self):
        self.previous_weights = self.model.trainable_weights
        # Pre Gradient Application
        for function in self.pre_gradient_application:
            function()
        self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))
        # Post Gradient Application
        for function in self.post_gradient_application:
            function()

    

    def evaluate(self, x, y):
        self.weight_zeros()
        # Pre Testing Forward Propagation
        for function in self.pre_evaluation:
            function()
        for x_step, y_step in zip(x,y):
            # Pre Inference
            for function in self.pre_inference:
                function()
            try: # to append the next prediction vector onto logits
                logits = tf.concat([logits,self.model(tf.expand_dims(x_step, axis=0),tf.expand_dims(y_step, axis=0))],0)
            except: # when logits doesn't exist, initialize it
                logits = self.model(tf.expand_dims(x_step, axis=0),tf.expand_dims(y_step, axis=0))
            # Post Inference
            for function in self.post_inference:
                function()
        # Post Testing Forward Propagation
        for function in self.post_evaluation:
            function()
        loss = self.loss_function(y, logits)
        one = tf.argmax(logits,1)
        two = tf.cast(y,one.dtype)
        accuracy = tf.math.count_nonzero(tf.math.equal(one,two)) / len(y)
        return [loss.numpy(),accuracy.numpy()]

    def add_test_metrics_entry(self, loss, accuracy, header):
        try:
            self.history.test.loss[header]
            self.history.test.loss[header] = {header:[loss]} 
        except:
            self.history.test.loss = self.history.test.loss.append(pd.DataFrame({header:[loss]}),ignore_index=True)
        self.history.test.accuracy = self.history.test.accuracy.append(pd.DataFrame({header:[accuracy]}),ignore_index=True)
        self.history.test.loss.to_csv(self.history.testDir + "/Loss.csv")
        self.history.test.accuracy.to_csv(self.history.testDir + "/Accuracy.csv")
        

    def fit(self, epochs, train_dataset, variant_iteration=0):
        self.weight_zeros()
        # Post Initialization
        for function in self.post_initialization:
            function()
        # Baseline Evaluation
        (loss,accuracy) = self.evaluate()
        self.test_loss       = [loss]
        self.test_accuracy   = [accuracy]

        # Console output
        print("--------------------------")
        print(self.header[variant_iteration])
        print("Baseline\tLoss: ", "{:.4f}".format(loss), "\tAccuracy: ", "{:.2f}".format(accuracy*100),"%")

        # Fitting
        for epoch in range(epochs):
            with IncrementalBar('Epoch ' + str(epoch), max=len(train_dataset), suffix="%(index)d/%(max)d - %(eta)ds\tLoss: " + "{:.4f}".format(loss) + "\tAccuracy: " + "{:.2f}".format(accuracy*100) + "%%") as bar:
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.grads = self.step(tf.constant(x_batch_train), tf.constant(y_batch_train))
                    self.apply_gradients()
                    bar.next()
                (loss,accuracy) =  self.evaluate()
                self.test_loss.append(loss)
                self.test_accuracy.append(accuracy)
        print("\tFinal loss: ", loss, "\tAccuracy: ", accuracy*100,"%")
        tf.keras.backend.clear_session()
        del self.model
        del self.optimizer
        del self.loss_function