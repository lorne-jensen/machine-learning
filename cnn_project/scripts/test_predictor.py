import torch
from src.predictor import predictor_test
from src.helpers import plot_confusion_matrix


# Load using torch.jit.load
model_reloaded =  torch.jit.load('checkpoints/original_exported.pt')

pred, truth = predictor_test(data_loaders['test'], model_reloaded)

plot_confusion_matrix(pred, truth)