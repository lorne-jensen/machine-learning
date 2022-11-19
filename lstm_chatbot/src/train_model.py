import random

import torch
from torch import optim, nn


#########################
# taken from https://www.guru99.com/seq2seq-model.html
#########################
from src.data_to_tensors import batch_to_train_data
from src.prepare_data import tensor_from_pair
from src.train import maskNLLLoss

teacher_forcing_ratio = 0.5


def clacModel(model, input_tensor, target_tensor, model_optimizer, mask):
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
        iter_loss, nTotal = maskNLLLoss(output[ot], target_tensor[ot], mask)
        loss += iter_loss

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss


def train_model(model, voc, pairs, batch_size=64, num_iteration=20000):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_batches = [batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(num_iteration)]

    for idx in range(1, num_iteration + 1):
        # training_pair = source[iter - 1]
        training_batch = training_batches[idx - 1]
        # Extract fields from batch
        source, lengths, target, mask, max_target_len = training_batch
        # input_tensor = source[idx]
        # target_tensor = target[idx]

        loss = clacModel(model, source, target, optimizer, mask)

        total_loss_iterations += loss

        if idx % 5000 == 0:
            avg_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (idx, avg_loss))

    torch.save(model.state_dict(), 'mytraining.pt')
    return model