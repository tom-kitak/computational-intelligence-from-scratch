import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import LossFunction as lf
import ActivationFunction as af

class Perceptron:
    
    def __init__(self, feature_vector_size, activation_function, learning_rate=0.05,
                 convergence_threshold=0.01, max_epochs=10000):
        """
        Initialize the Perceptron object with the provided parameters.

        Parameters
        ----------
        feature_vector_size : int
            The size of the feature vector.
        activation_function : function
            The activation function to be used by the Perceptron.
        learning_rate : float, default 0.05
            The learning rate of the Perceptron.
        convergence_threshold : float, default 0.01
            The threshold value for determining convergence of the Perceptron.
        max_epochs : int, default 10000
            The maximum number of epochs that the Perceptron can run before terminating.

        Returns
        -------
        None
        """
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.bias = np.random.random()
        self.weights = np.append(np.random.random(feature_vector_size), self.bias)

    def train(self, data, labels, operator):
        """
        Train the Perceptron on the provided data and labels.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be used for training.
        labels : array-like of shape (n_samples,)
            The corresponding labels for the data.
        operator : str
            The operator used to train the Perceptron. Either 'AND' or 'OR'.

        Returns
        -------
        None
        """
        data = np.array([np.append(i, 1) for i in np.array(data)])
        labels = np.array(labels)
        converged, epoch, costs = False, 0, []
        while not converged and epoch < self.max_epochs:
            for index, data_point in enumerate(data):
                if len(data_point) == len(data[0]) - 1:
                    data_point = np.append(data_point, [1])
                output = self.get_prediction(data_point)
                self.update_weights(data_point, output, labels[index])
            cost = self.calculate_cost(data, labels)
            costs.append(cost)
            converged = cost < self.convergence_threshold
            epoch += 1
        if converged:
            print(f"Converged in {epoch} epochs.")
        elif epoch >= self.max_epochs:
            print(f"Could not converge in {self.max_epochs} epochs.")
        self.plot_graph(costs, epoch, operator)

    def get_prediction(self, data_point):
        """
        Compute the predicted output for a single data point.

        Parameters
        ----------
        data_point : array-like of shape (n_features,)
            The data point for which the predicted output is to be computed.

        Returns
        -------
        output : float
            The predicted output for the data point.
        """
        z = np.dot(self.weights, data_point)
        output = self.activation_function(z)
        return output

    def update_weights(self, data_point, predicted_output, actual_output):
        """
        Update the weights based on the prediction and actual output.

        Parameters
        ----------
        data_point : array-like of shape (n_features,)
            The data point for which the weights are to be updated.
        predicted_output : float
            The predicted output for the data point.
        actual_output : float
            The actual output for the data point.

        Returns
        -------
        None
        """
        for i in range(len(self.weights)):
            error = actual_output - predicted_output
            delta_w = self.learning_rate * data_point[i] * error
            self.weights[i] += delta_w

    def calculate_cost(self, data, labels):
        """
        Calculate the mean squared error (MSE) of the predicted outputs and the actual labels.

        Parameters
        ----------
        data : numpy.ndarray
            An array of feature vectors of shape (n_samples, n_features).
        labels : numpy.ndarray
            An array of labels of shape (n_samples,).

        Returns
        -------
        float
            The mean squared error of the predicted outputs and the actual labels.

        """
        outputs = [self.get_prediction(x) for x in data]
        return (1 / len(data)) * np.sum(np.square(labels - outputs))
    
    def plot_graph(self, costs, epochs, operator):
        """
        Generate a plot of the error (MSE) over epochs for the Perceptron.

        Parameters
        ----------
        costs : list
            A list of the costs for each epoch.
        epochs : int
            The number of epochs the Perceptron ran for.
        operator : str
            The type of logical operator the Perceptron is learning.

        """
        plt.rc('font', family='serif', size=16)
        plt.rc('lines', markersize=6)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc("figure", figsize=(6, 6))
        
        ax = plt.subplot()
        ax.set_ylabel("Error (MSE)", labelpad=8)
        ax.set_xlabel("Epoch", labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(range(epochs), costs)
        ax.set_title("Error (MSE) over epochs for "+ operator, y=1.05)
        plt.tight_layout()
        plt.savefig(f"./docs/plots/{operator}_error_over_epochs")
        plt.show()

class ANN:
    def __init__(self, perceptrons_per_layer, loss_function=lf.CategoricalCrossEntropy, activation_function_output_layer=af.Softmax, activation_function_hidden_layer=af.Sigmoid, num_of_epochs=20, learning_rate=0.05, mini_batch_size=20, weights_initialisation=np.random.standard_normal):
        """
        Constructor for Artificail Neural Network.
        
        weights : list of matrices
            Initialised from self.weights_initialisation distribution
        biases : list of vectors
            Initialised from self.weights_initialisation distribution

        Parameters
        ----------
        perceptrons_per_layer : list
            List indicating number of layers and perceptrons per layer.
            For example if perceptrons_per_layer = [5, 10, 20, 3] then this indicates 4 layers,
            with 5 perceptrons in (first) input layer, 3 perceptrons in (last) output layer, and
            10 and 20 perceptrons in the (middle) hidden layer respectively.
        loss_function : function
            Function to calculate Loss
        activation_function_output_layer : function
            Function to be used as activation function for the output layer.
            Needs option to return the derivative.
        activation_function_hidden_layer : function
            Function to be used as activation function for the hidden layer.
            Needs option to return the derivative.
        num_of_epochs : int
            Number to indicate the number of epochs to perform while training 
        learning_rate : int
            Learning rate to be used while training 
        mini_batch_size : int
            Size of the mini batch
        weights_initialisation : function, by defauly standard_normal
            Functions that samples the desired distribution for initialising weights and biases
        """
        self.perceptrons_per_layer = perceptrons_per_layer
        self.layer_count = len(perceptrons_per_layer)
        self.loss_function = loss_function
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.weights_initialisation = weights_initialisation
        self.activation_function_output_layer = activation_function_output_layer
        self.activation_function_hidden_layer = activation_function_hidden_layer

        # Draw samples from a standard Normal distribution
        self.weights = []
        self.biases = []
        for (row, col) in zip(perceptrons_per_layer[1:], perceptrons_per_layer[:-1]):
            self.weights.append(weights_initialisation(size=(row, col))) 
        for row in perceptrons_per_layer[1:]:
            self.biases.append(weights_initialisation(size=(row, 1)))

    def plot_performance_two_datasets(self, validation_data, test_data):

        x = range(1, len(validation_data)+1)
        plt.xticks(x)
        plt.plot(x, validation_data, color='r', label='Validation Data')
        plt.plot(x, test_data, color='g', label='Test Data')
  
        plt.xlabel("Epoch")
        plt.ylabel("Performance")
        plt.title("Validation Data vs Test Data performance")
        
        plt.legend()
        plt.show()

    def evaluate(self, validation_data):
        """
        Evaluates the ANN model by calculating accuracy.

        Parameters
        ----------
        validation_data : list of tuples (instance_features, instance_label)
            data to compute the prediction accuracy of
        
        Returns
        ----------
        float
            Prediction accuracy
        """
        correct_count = 0

        for instance_features, instance_label in validation_data:
            prediction_output = self.forward_pass(instance_features)
            # Gets the prediciton label by extracting which last layer perceptron is most activated
            predicted_label = 0
            max_activation = prediction_output[0]
            for i in range(1, len(prediction_output)):
                if prediction_output[i] > max_activation:
                    max_activation = prediction_output[i]
                    predicted_label = i
            true_label = 0
            max_value = instance_label[0]
            for i in range(1, len(instance_label)):
                if instance_label[i] > max_value:
                    max_value = instance_label[i]
                    true_label = i
            if predicted_label == true_label:
                correct_count += 1

        return correct_count / len(validation_data)
    
    def forward_pass(self, input_instance):
        """
        Output prediction for input instance.

        Parameters
        ----------
        input_instance : vector
            vector to compute the prediction of

        Returns
        ----------
        predicted label
        """
        # Iterate over weights and biases separately
        for layer in range(len(self.weights) - 1):
            # Compute output of current layer
            w = self.weights[layer]
            b = self.biases[layer]
            input_instance = self.activation_function_hidden_layer.activation(np.dot(w, input_instance) + b)

        # Last layer could have different activation function
        w = self.weights[-1]
        b = self.biases[-1]
        input_instance = self.activation_function_output_layer.activation(np.dot(w, input_instance) + b)

        return input_instance

    def train(self, train_data, validation_data=None, test_data=None, plot_performance=False):
        """
        Trains the neural network.

        Parameters
        ----------
        train_data : list of tuples (features, label)
            data to train the neural network with
        """
        performance_validation_data = []
        performance_test_data = []

        previousEpochValidationSum = 10000000

        for epoch in range(self.num_of_epochs):

        # First shuffle the data at the start of every epoch
            np.random.shuffle(train_data)

            # Split the data into mini batches
            mini_batches = []
            for batch in range(0, len(train_data), self.mini_batch_size):
                mini_batch = train_data[batch:batch + self.mini_batch_size]
                mini_batches.append(mini_batch)

            # Use mini batch to make a small step into more optimal direction
            for mini_batch in mini_batches:
                self.update_network_parameters(mini_batch)
            #print(f"Epoch number:", epoch)

            if (plot_performance and validation_data is not None and test_data is not None):
                performance_validation_data.append(self.evaluate(validation_data))
                performance_test_data.append(self.evaluate(test_data))

            newEpochValidationSum = self.getValidationTotalLoss(validation_data)


            if (newEpochValidationSum >= previousEpochValidationSum
                    or newEpochValidationSum * 1.01 > previousEpochValidationSum):
                print("Early stopping!!!!!!!!")
                break
            else:
                print("Epoch number " + str(epoch))

            previousEpochValidationSum = newEpochValidationSum
        
        if (plot_performance and validation_data is not None and test_data is not None):
            self.plot_performance_two_datasets(performance_validation_data, performance_test_data)

    def update_network_parameters(self, mini_batch):
        """
        Making a step in more optimal direction according to mini_batch
        using gradient descent and backpropagation.

        Parameters
        ----------
        mini_batch : list of tuples (instance_features, instance_label)
        """
        weight_gradients = []
        bias_gradients = []
        for weight in self.weights:
            weight_gradients.append(np.zeros(weight.shape))
        for bias in self.biases:
            bias_gradients.append(np.zeros(bias.shape))

        for instance_features, instance_label in mini_batch:
            # For every instance in the mini batch calculate the gradient,
            # which indicates which changes to the ANN single instance wants to make.
            delta_weight_gradients, delta_bias_gradients = self.backpropagation(instance_features, instance_label)
            
            # Now we add gradients or "wants" of all instances 
            # to get what change to the ANN is best overall.
            for i in range(len(weight_gradients)):
                weight_gradients[i] = weight_gradients[i] + delta_weight_gradients[i]
            for i in range(len(bias_gradients)):
                bias_gradients[i] = bias_gradients[i] + delta_bias_gradients[i]

        # Updating the weighs according to calculated gradient over instanes in the mini_batch
        # W_(t+1) = W_(t) - (proportional)learning_rate * derivative_loss_with_respect_to_weights
        for i in range(len(weight_gradients)):
            self.weights[i] = self.weights[i] - (self.learning_rate / len(mini_batch)) * weight_gradients[i]
        # W_(t+1) = W_(t) - (proportional)learning_rate * derivative_loss_with_respect_to_biases
        for i in range(len(bias_gradients)):
            self.biases[i] = self.biases[i] - (self.learning_rate / len(mini_batch)) * bias_gradients[i]

    def backpropagation(self, instance_features, instance_label):
        """
        Calculate weight_gradients, bias_gradients for a single instance

        Parameters
        ----------
        instance_features : vector
            vector of feature values of the instance
        instance_label : int
            label of the instance
        
        Returns
        ----------
        weight_gradients : list of matrices
        bias_gradients : list of vectors
        """
        weight_gradients = []
        bias_gradients = []
        for weight in self.weights:
            weight_gradients.append(np.zeros(weight.shape))
        for bias in self.biases:
            bias_gradients.append(np.zeros(bias.shape))

        activation_input = instance_features
        z_values = []
        activations = [activation_input]

        # Feedforward to obtain activations and outputs z
        for i in range(len(self.weights) - 1):
            weight = self.weights[i]
            bias = self.biases[i]
            z = np.dot(weight, activation_input) + bias
            z_values.append(z)
            activation_input = self.activation_function_hidden_layer.activation(z)
            activations.append(activation_input)

        # Last layer could use different actication function
        weight = self.weights[-1]
        bias = self.biases[-1]
        z = np.dot(weight, activation_input) + bias
        z_values.append(z)
        activation_input = self.activation_function_output_layer.activation(z)
        activations.append(activation_input)

        # Backpropagation calculating last layer
        if (type(self.loss_function) is type(lf.CategoricalCrossEntropy) and type(self.activation_function_output_layer) is type(af.Softmax)):
            # based on https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
            # and also based on https://themaverickmeerkat.com/2019-10-23-Softmax/
            # Computing deivative L with respect to z
            reusable_error = activations[-1] - instance_label
        else:
        # (derviative L with respect to a) * (derivative a wrt z) are reusable
            d_L_wrt_a = self.loss_function.derivative(activations[-1], instance_label)
            d_a_wrt_z = self.activation_function_output_layer.derivative(z_values[-1])
            reusable_error = d_L_wrt_a * d_a_wrt_z

        
        
        # Bias gradient of the last layer is just "reusable_error" since derivative_z_with_respect_to_b is 1
        bias_gradients[-1] = reusable_error
        # Weight gradient of last layer is (reusable * derivative_z_with_respect_to_w)
        # (reusable_error * derivative_z_with_respect_to_w) = reusable_error * activation of the penultimate layer 
        weight_gradients[-1] = np.dot(reusable_error, activations[-2].T)
        
        # Propagating the error back from the penultimate layer to the first layer
        for layer in range(2, self.layer_count):
            d_a_wrt_z = self.activation_function_hidden_layer.derivative(z_values[-layer])
            # derivative L with respect to a of the current layer
            d_L_wrt_a = np.dot(self.weights[-layer+1].T, reusable_error)
            reusable_error =  d_L_wrt_a * d_a_wrt_z

            # Bias gradient of the last layer is just "reusable error" since derivative_z_with_respect_to_b is 1
            bias_gradients[-layer] = reusable_error
            # Weight gradient of last layer is (reusable_error * derivative_z_with_respect_to_w)
            # (reusable_error * derivative_z_with_respect_to_w) = reusable_error * activation of the penultimate layer 
            weight_gradients[-layer] = np.dot(reusable_error, activations[-layer-1].T)

        return weight_gradients, bias_gradients
    
    def getValidationTotalLoss(self , validationSet):
        totalSum = 0
        for elem in validationSet:
            prediction = self.forward_pass(elem[0])
            loss = self.loss_function.loss(prediction, elem[1])
            totalSum = totalSum + loss

        return totalSum