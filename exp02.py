import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics


class NeuralNet:
    def __init__(self, input_size, output_size):
        """
        The constructor. Initialize the architecture and structural
        components of the training algorithm (e.g., which loss function and optimization strategy
        to use)
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        """
        
        self.architecture = self._initialize_architecture(input_size=input_size, output_size=output_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.architecture.parameters())

    @staticmethod
    def _initialize_architecture(input_size=28*28, output_size=10):
        """
        This private method instantiates the overarching architecture of the neural network.
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        :return: The overarching architecture of the network.
        """
        
        architecture = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),  # Apply ReLU activation to previous layer.
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),  # Apply ReLU activation to previous layer.
            nn.Linear(in_features=64, out_features=output_size)
        )
        return architecture

    @staticmethod
    def _convert_to_one_hot_encoding(y):
        """
        Converts digit to a 1 hot encoded value.

        The mapping is as follows:
        0 -> [1, 0, 0, 0, ..., 0]
        1 -> [0, 1, 0, 0, ..., 0]
        2 -> [0, 0, 1, 0, ..., 0]
        ...
        9 -> [0, 0, 0, 0, ..., 1]

        :param y: The scalar value(s).
        :return: The 1 hot encoded vector(s).
        """
        
        result = np.zeros(10)
        result[y] = 1

        return result

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
        
        # Set a default that the batch size is sqrt(# of training instances)
        if batch_size is None:
            batch_size = int(np.sqrt(len(x)))

        # before converting y to a tensor, turn all y-values into 1 hot code
        y = onehotencode(y)

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
                # loss = self.loss_function(pred_minibatch.squeeze(dim=1), y_minibatch)
                loss = self.loss_function(pred_minibatch.squeeze(dim=1), y_minibatch.squeeze(dim=1))

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
        
        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        # Let the model know we're in "evaluate" mode (as opposed to training)
        self.architecture.eval()
        # Feed through the architecture.
        result = self.architecture(x)

        result = torch.argmax(result, dim=1)

        # Convert back to numpy
        result = result.detach().numpy()

        return result

class Exp02:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        """
        x_train, y_train, x_test, y_test = None, None, None, None

        
        # opening the CSV file
        train = np.loadtxt(file_path_prefix + "mnist_train.csv", delimiter=',')
        test = np.loadtxt(file_path_prefix + "mnist_test.csv", delimiter=',')
        x_train = train[:, 1:]
        y_train = train[:, 0]

        x_test = test[:, 1:]
        y_test = test[:, 0]

        # change y values to 1 hot encode
        # y_train, y_test = onehotencode(y_train), onehotencode(y_test)

        return x_train, y_train, x_test, y_test


    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp02.load_train_test_data()

        print("Training Model...")

        #######################################################################
        # Initialize, Train, Evaluate.
        #######################################################################

        # (1) Initialize model;

        # Number of inputs == the number of features
        number_of_inputs = len(x_train[0])
        # Number of outputs == 1; Only outputing 10 variables
        number_of_outputs = 10

        model = NeuralNet(input_size=number_of_inputs, output_size=number_of_outputs)

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'
        model.fit(x_train, y_train)

        print("Training complete!")
        print()

        print("Evaluating Model")
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        print("-- SciKit Learn Classification Report: Training Data")
        print(metrics.classification_report(y_train, y_train_pred))

        print("-- SciKit Learn Classification Report: Training Data")
        print(metrics.classification_report(y_test, y_test_pred))

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)

def onehotencode(y_arr):
    y_temp = []
    for y_i in y_arr:
        result = NeuralNet._convert_to_one_hot_encoding(int(y_i))
        y_temp.append(np.asarray(result))
    # return np array to update y
    return np.array(y_temp)

