import random

import torch
from torch import optim, nn


#########################
# taken from https://www.guru99.com/seq2seq-model.html
#########################

from src.prepare_data import tensor_from_pair

teacher_forcing_ratio = 0.5


def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    # print(input_tensor.shape)

    output = model(input_tensor, target_tensor)

    num_iter = output.size(0)
    print(num_iter)

    # calculate the loss from a predicted sentence with the expected result
    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss


def train_model(model, source, target, pairs, num_iteration=20000):
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    # training_pairs = [tensor_from_pair(voc, random.choice(pairs)) for i in range(num_iteration)]

    training_pairs = [(random.choice(source), random.choice(target)) for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = clacModel(model, input_tensor, target_tensor, optimizer, criterion)

        total_loss_iterations += loss

        if iter % 5000 == 0:
            avarage_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, avarage_loss))

    torch.save(model.state_dict(), 'mytraining.pt')
    return model