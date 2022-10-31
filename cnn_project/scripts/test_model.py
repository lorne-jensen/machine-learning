# load the model that got the best validation accuracy
from src.train import one_epoch_test
from src.data import get_data_loaders
from src.optimization import get_loss
from src.model import MyModel
import torch


batch_size = 128        # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2       # fraction of the training data to reserve for validation
num_epochs = 50        # number of epochs for training
num_classes = 50       # number of classes. Do not change this
dropout = 0.01          # dropout for our model
learning_rate = 0.00025  # Learning rate for SGD (or Adam)
opt = 'adam'            # optimizer. 'sgd' or 'adam'
weight_decay = 0.0     # regularization. Increase this to combat overfitting
momentum = 0.9

loss = get_loss()

data_loaders = get_data_loaders(batch_size=batch_size, valid_size=valid_size)

model = MyModel(num_classes=num_classes, dropout=dropout)

model.load_state_dict(torch.load("checkpoints/best_val_loss.pt"))

# Run test
one_epoch_test(data_loaders['test'], model, loss)
