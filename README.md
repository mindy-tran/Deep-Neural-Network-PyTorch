# Deep-Neural-Network-PyTorch
Machine Learning project: Implement two deep feed-forward neural networks in PyTorch, treating the identification as a regression problem and a classification problem.

Mindy Hoang Tran, Completed March 2023

Implemented and compared multiple variations of a neural network using a widely
used library.
1. Constructed a deep feed-forward neural network in PyTorch which identifies images in the
MNIST dataset. This model will treat the identification as a regression problem.
2. Constructed a deep feed-forward neural network in PyTorch which identifies images in the
MNIST dataset. This model will treat the identification as a classification problem.
3. Compare all of the implemented models and provide summary statistics about their
efficacy.


*********************************************************************************
Task 0, Set up:

1. Install the required libraries for this project by running the following commands:

pip install scikit-learn (Or, for Anaconda conda install scikit-learn)

pip install scipy (Or, for Anaconda conda install scipy)

pip install numpy (Or, for Anaconda conda install numpy)

pip install matplotlib (Or, for Anaconda conda install matplotlib)

2. Install Pytorch. The following commands will do:
pip install torch torchvision torchaudio
OR
conda install pytorch torchvision torchaudio cpuonly -c pytorch

3. Unzip the mnist_test_train.zip into your directory to get the mnist_test.csv and mnist_train.csv files.


*********************************************************************************
Task 1:  Run main.py with exp = Exp01() uncommented in the main function

This is a deep feed-forward neural network (regression) in PyTorch. This model will treat the identification as a regression problem.
The neural network has two linear feed-forward layers of size 128 and 64 nodes, and uses ReLU activation functions. The network trains for 25 epochs.
I used Adam Optimization for my optimizer and Mean Squared Error for my loss function.

*********************************************************************************
Task 2:  Run main.py with exp = Exp02() uncommented in the main function

This is a deep feed-forward neural network (classification) in PyTorch. This model will treat the identification as a classification problem.
The network uses softmax values and 1-hot encoding. The neural network has two linear feed-forward layers of size 128 and 64 nodes, and uses ReLU activation functions. The network trains for 25 epochs.
I used Adam Optimization for my optimizer and Categorical Cross Entropy for my loss function.
