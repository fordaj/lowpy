import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import pandas as pd
import datetime
from progress.bar import IncrementalBar
from progress.bar import Bar

class wrapper:
    def __init__(self,
        metrics,
        variability_stdev=0.0, 
        decay=1.0, 
        precision=0, 
        upper_bound=0.1, 
        lower_bound=-0.1, 
        percent_stuck_at_lower_bound=0, 
        percent_stuck_at_zero=0, 
        percent_stuck_at_upper_bound=0,
        rtn_stdev=0,
        drift_rate_to_upper=0,
        drift_rate_to_zero=0,
        drift_rate_to_lower=0,
        drift_rate_to_bounds=0
    ):
        self.metrics = metrics
        self.variability_stdev = variability_stdev
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
        self.weight_zeros()

    def plot(self,varied_parameter):
        self.header = varied_parameter

    def weight_zeros(self):
        weights = self.model.trainable_weights
        self.zeros = []
        for w in weights:
            self.zeros.append(tf.Variable(tf.zeros(w.shape,dtype=tf.dtypes.float32)))
        self.cell_updates = self.zeros

    def initialization_variability(self):
        weights = []
        for l in self.model.layers:
            for w in range(len(l.weights)):
                if (not 'conv' in l.weights[w].name) and (not 'embed' in l.weights[w].name):
                    weights.append(tf.random.normal(l.weights[w].shape,mean=l.weights[w],stddev=self.variability_stdev))

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
                weights[w].assign(tf.random.normal(weights[w].shape,mean=weights[w],stddev=self.variability_stdev))
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

    def track_weight_updates(self):
        for g in range(len(self.grad)):
            self.cell_updates[g].assign_add(tf.cast(self.grad[g] != 0, self.cell_updates[g].dtype))

    def step(self, x_step, y_step):
        """
        Training step function.
        1.) pre_training_forward_propagation
        2.) Forward Propagation
        3.) post_training_forward_propagation
        4.) pre_gradient_calculation
        5.) Gradient Calculation
        6.) post_gradient_calculation
        """

        # Pre Training Forward Propagation
        for function in self.pre_train_forward_propagation:
            function()

        # Forward Propagation
        with tf.GradientTape() as tape:
            logits = self.model(x_step, training=True)
            loss_value = self.loss_function(y_step, logits)

        # Post Training Forward Propagation
        for function in self.post_train_forward_propagation:
            function()

        # Pre Gradient Calculation
        for function in self.pre_gradient_calculation:
            function()

        # Gradient Calculation
        self.grad = tape.gradient(loss_value, self.model.trainable_weights)

        # Post Gradient Calculation
        for function in self.post_gradient_calculation:
            function()
    
    def apply_gradients(self):
        """
        Gradient application function.
        1.) pre_gradient_application
        2.) Gradient Application
        3.) post_gradient_application
        """

        # Pre Gradient Application
        for function in self.pre_gradient_application:
            function()

        self.optimizer.apply_gradients(zip(self.grad, self.model.trainable_weights))

        # Post Gradient Application
        for function in self.post_gradient_application:
            function()

    def evaluate(self, x, y, batch_size = 1, verbose = 1):
        # Batch dataset
        evaluation_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        
        # Pre Testing Forward Propagation
        for function in self.pre_evaluation:
            function()

        with Bar('Evaluation:\t', max=len(evaluation_dataset)) as bar:
            for step, (x_step, y_step) in enumerate(evaluation_dataset):
                # Pre Inference
                for function in self.pre_inference:
                    function()
                
                try: # to append the next prediction vector onto logits
                    logits = tf.concat([logits,self.model(x_step)],0)
                except: # when logits doesn't exist, initialize it
                    logits = self.model(x_step)
                
                # Post Inference
                for function in self.post_inference:
                    function()
                bar.next()
        
        # Post Evaluation
        for function in self.post_evaluation:
            function()
        y = np.concatenate([y for x, y in evaluation_dataset], axis=0)
        loss = self.loss_function(y, logits)
        winningNeurons = tf.argmax(logits,1)
        labels = tf.cast(y,winningNeurons.dtype)
        accuracy = tf.math.count_nonzero(tf.math.equal(winningNeurons,labels)) / len(y)
        return [loss.numpy(),accuracy.numpy()]

    def fit(self, x = None, y = None, batch_size = 1, epochs = 10, variant_number = 0, verbose = 1, validation_split = 0, validation_data = None, validation_batch_size = 1, shuffle = True):
        """
        Trains a model with nonidealities on a dataset over a specified number of epochs.

        Arguments
        x: Training data in the form of a matrix.
        y: Target data (labels) in the form of a vector.
        batch_size: Number of inputs per batch of training. Defaults to 1.
        epochs: Number of epochs before fitting process ends. Defaults to 10.
        variant_number: If simulating multiple model variants, represents the index of the model simulated.
        verbose: Descriptivity of the fitting process. 1 = console logging, 0 = none.
        validation_split: Float between 0 and 1 representative of the portion of the training dataset to be split for validation.
        validation_data: Validation data for model fitness evaluation. Should be a list (x_val, y_val). Overrides validation split.
        validation_batch_size: Number of inputs per batch of validation. Defaults to 1.
        shuffle: Defaults to True. Shuffles training dataset
        """

        # Split into validation and training datasets
        numInputs = len(y)
        if (validation_data is None):
            numValidation   = round(numInputs*validation_split)
            numTrain        = numInputs - numValidation
            x_validation    = x[numTrain:]
            y_validation    = y[numTrain:]
        else:
            numTrain        = numInputs
            x_validation    = validation_data[0]
            y_validation    = validation_data[1]
        x_train = x[:numTrain]
        y_train = y[:numTrain]
        train_dataset           = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        # Batch training dataset and shuffle (if applicable)
        if (shuffle):
            train_dataset       = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        else:
            train_dataset       = train_dataset.batch(batch_size)
        
        # Simulate post initialization nonidealities
        for function in self.post_initialization:
            function()

        # Baseline validation
        print("--------------------------")
        (loss,accuracy) = self.evaluate(x=x_validation, y=y_validation)
        test_loss       = [loss]
        test_accuracy   = [accuracy]
        print("Variant Number: ", str(variant_number))
        print("Baseline\tLoss: ", "{:.4f}".format(loss), "\tAccuracy: ", "{:.2f}".format(accuracy*100),"%")
        self.metrics.loss.add_value(loss, 0, variant_number)
        self.metrics.accuracy.add_value(accuracy, 0, variant_number)
        # Fitting
        for epoch in range(1,epochs+1):
            with IncrementalBar('Epoch ' + str(epoch) + '\t\t', max=len(train_dataset), suffix="%(index)d/%(max)d - %(eta)ds\tLoss: " + "{:.4f}".format(loss) + "\tAccuracy: " + "{:.2f}".format(accuracy*100) + "%%") as bar:
                for step, (x_step, y_step) in enumerate(train_dataset):
                    self.step(tf.constant(x_step), tf.constant(y_step))
                    self.apply_gradients()
                    bar.next()
                print("")
                (loss,accuracy) =  self.evaluate(x=x_validation, y=y_validation)
                self.metrics.loss.add_value(loss, epoch, variant_number)
                self.metrics.accuracy.add_value(accuracy, epoch, variant_number)
        print("\tFinal loss: ", loss, "\tAccuracy: ", accuracy*100,"%")