import torch

from src.data import get_data_loaders
from src.predictor import predictor_test
from src.helpers import plot_confusion_matrix


batch_size = 64  # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2  # fraction of the training data to reserve for validation
num_epochs = 35  # number of epochs for training
num_classes = 50  # number of classes. Do not change this
learning_rate = 0.001  # Learning rate for SGD (or Adam)
opt = 'adam'      # optimizer. 'sgd' or 'adam'
weight_decay = 0.0 # regularization. Increase this to combat overfitting

model_reloaded = torch.jit.load("checkpoints/transfer_exported.pt")
data_loaders = get_data_loaders(batch_size=batch_size)

pred, truth = predictor_test(data_loaders['test'], model_reloaded)

plot_confusion_matrix(pred, truth)
