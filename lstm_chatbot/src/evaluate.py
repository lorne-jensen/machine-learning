import random

import torch

from src.prepare_data import tensor_from_sentence, MAX_LENGTH
from src.vocab import EOS_token, Voc


#########################
# modified from https://www.guru99.com/seq2seq-model.html
#########################

def evaluate(model, input_question, output_answer, sentences, voc: Voc, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_question, sentences[0])
        output_tensor = tensor_from_sentence(output_answer, sentences[1])

        decoded_words = []

        output = model(input_tensor, output_tensor)
        # print(output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            # print(topi)

            if topi[0].item() == EOS_token:
                decoded_words.append(voc.index2word(EOS_token))
                break
            else:
                decoded_words.append(output_answer.index2word[topi[0].item()])
    return decoded_words


def evaluate_randomly(model, source, target, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('source\n {}'.format(pair[0]))
        print('target \n{}'.format(pair[1]))
        output_words = evaluate(model, source, target, pair)
        output_sentence = ' '.join(output_words)
        print('predicted\n{}'.format(output_sentence))