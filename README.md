# Deep-Neural-Network-PyTorch
Machine Learning project: Implement two deep feed-forward neural networks in PyTorch, treating the identification as a regression problem and a classification problem.

Mindy Tran
Project for Machine Learning for Data Science, Winter 2023

In this project you will implement and compare multiple variations of a neural network using a widely
used library.
1. You will construct a deep feed-forward neural network in PyTorch which identifies images in the
MNIST dataset following an explicit blueprint provided in the instructions. This model will treat
the identification as a regression problem, to make it most comparable to the linear regression
you did in the previous project.
2. You will construct a deep feed-forward neural network in PyTorch which identifies images in
the MNIST dataset following an explicit blueprint provided in the instructions. This network will
be a slight modification of the network created in 2(a), instead treating the identification as a
classification problem.
3. You will compare all of the implemented models and provide summary statistics about their
efficacy.


*********************************************************************************
Task 0, Set up:

1. Install the required libraries for this project by running the following commands:
pip install scikit-learn (Or, for Anaconda conda install scikit-learn)
pip install scipy (Or, for Anaconda conda install scipy)
pip install numpy (Or, for Anaconda conda install numpy)
pip install matplotlib (Or, for Anaconda conda install matplotlib)

2. Install Pytorch. We think the following commands will do:
pip install torch torchvision torchaudio
OR
conda install pytorch torchvision torchaudio cpuonly -c pytorch

3. Unzip the mnist_test_train.zip into your directory


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
