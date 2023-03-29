import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class NeuralNet:

    def __init__(self, input_size, output_size):
        """
        The constructor. This is where you should initialize the architecture and structural
        components of the training algorithm (e.g., which loss function and optimization strategy
        to use)
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        """
        # Complete me!
        self.architecture = self._initialize_architecture(input_size=input_size, output_size=output_size)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.architecture.parameters())

    @staticmethod
    def _initialize_architecture(input_size=28*28, output_size=1):
        """
        This private method instantiates the overarching architecture of the neural network.
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        :return: The overarching architecture of the network.
        """
        # Complete me!
        architecture = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),  # Apply ReLU activation to previous layer.
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),  # Apply ReLU activation to previous layer.
            nn.Linear(in_features=64, out_features=output_size)
        )
        return architecture

    def fit(self, x, y, epochs=25, batch_size=None, verbose=True):
        """
        Train the model to predict y when given x.
        :param x: The input/features data
        :param y: The the output/target data
        :param epochs: The number of times to iterate over the training data.
        :param batch_size: How many samples to consider at a time before updating network weights.
        sqrt(# of training instances) by default.
        :param verbose: Print a lot of info to the console about training.
        :return: None. Internally the weights of the network are updated.
        """
        # Complete me!
        # Set a default that the batch size is sqrt(# of training instances)
        if batch_size is None:
            batch_size = int(np.sqrt(len(x)))

        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Train for the given number of epochs.
        for epoch in range(epochs):
            if verbose:
                print("Working on Epoch", epoch, "of", epochs)

            # Let the model know we're starting another round of training
            self.architecture.train()

            # Organize into helper methods which automatically put things into
            # well formed batches for us.
            train_ds = TensorDataset(x, y)
            train_dl = DataLoader(train_ds, batch_size=batch_size)

            # Iterate over each minibatch
            for x_minibatch, y_minibatch in train_dl:
                # Feed through the architecture
                pred_minibatch = self.architecture(x_minibatch)

                # Calculate the loss
                loss = self.loss_function(pred_minibatch.squeeze(dim=1), y_minibatch)

                # Apply back propagation using the specified optimizer
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if verbose:
                print("Loss at end of epoch:", loss)

    def predict(self, x):
        """
        Predict an outcome given a sample of features.
        :param x: The features to use for a prediction.
        :return: The predicted outcome based on the features.
        """
        # Complete me!
        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        # Let the model know we're in "evaluate" mode (as opposed to training)
        self.architecture.eval()
        # Feed through the architecture.
        result = self.architecture(x)
        # Convert back to numpy
        result = result.detach().numpy()
        return result


class Exp01:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        """

        x_train, y_train, x_test, y_test = None, None, None, None

        # Fix & Complete me
        # opening the CSV file
        train = np.loadtxt(file_path_prefix + "mnist_train.csv", delimiter=',')
        test = np.loadtxt(file_path_prefix + "mnist_test.csv", delimiter=',')
        x_train = train[:, 1:]
        y_train = train[:, 0]

        x_test = test[:, 1:]
        y_test = test[:, 0]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def compute_mean_absolute_error(true_y_values, predicted_y_values):
        list_of_errors = []
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            error = abs(true_y - pred_y)
            list_of_errors.append(error)
        mean_abs_error = np.mean(list_of_errors)
        return mean_abs_error

    @staticmethod
    def compute_mean_error_rate(true_y_values, predicted_y_values):
        dif = 0
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            if true_y != int(np.round(pred_y)):
                dif += 1
        mean_error_rate = dif / len(true_y_values)
        return mean_error_rate

    @staticmethod
    def print_error_report(trained_model, x_train, y_train, x_test, y_test):
        print("\tEvaluating on Training Data")
        # Evaluating on training data is a less effective as an indicator of
        # accuracy in the wild. Since the model has already seen this data
        # before, it is a less realistic measure of error when given novel/unseen
        # inputs.
        #
        # The utility is in its use as a "sanity check" since a trained model
        # which preforms poorly on data it has seen before/used to train
        # indicates underlying problems (either more data or data preprocessing
        # is needed, or there may be a weakness in the model itself.

        y_train_pred = trained_model.predict(x_train)

        mean_absolute_error_train = Exp01.compute_mean_absolute_error(y_train, y_train_pred)

        mean_error_rate_train = Exp01.compute_mean_error_rate(y_train, y_train_pred)

        print("\tMean Absolute Error (Training Data):", mean_absolute_error_train)
        print("\tMean Error Rate (Training Data):", mean_error_rate_train)
        print()

        print("\tEvaluating on Testing Data")

        y_test_pred = trained_model.predict(x_test)
        mean_absolute_error_test = Exp01.compute_mean_absolute_error(y_test, y_test_pred)
        mean_error_rate_test = Exp01.compute_mean_error_rate(y_test, y_test_pred)

        print("\tMean Absolute Error (Testing Data):", mean_absolute_error_test)
        print("\tMean Error Rate (Testing Data):", mean_error_rate_test)
        print()

    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp01.load_train_test_data()

        print("Training Model...")

        #######################################################################
        # Complete this 2-step block of code using the variable name 'model' for
        # the linear regression model.
        # You can complete this by turning the given psuedocode to real code
        #######################################################################

        # (1) Initialize model;

        # Number of inputs == the number of features
        number_of_inputs = len(x_train[0])
        # Number of outputs == 1; Only outputting 1 variable
        number_of_outputs = 1

        model = NeuralNet(input_size=number_of_inputs, output_size=number_of_outputs)

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'

        model.fit(x_train, y_train)

        print("Training complete!")
        print()

        print("Evaluating Model")
        Exp01.print_error_report(model, x_train, y_train, x_test, y_test)

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)
