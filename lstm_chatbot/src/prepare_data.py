import random
import re

import gensim
import nltk
import numpy as np
import pandas as pd
import gzip
import torch
import torchtext
import unicodedata
from nltk.corpus import brown

import os

from src.vocab import Voc, EOS_token

# nltk.download('brown')
# nltk.download('punkt')
#
# # Output, save, and load brown embeddings
#
# model = gensim.models.Word2Vec(brown.sents())
# model.save('brown.embedding')
#
# w2v = gensim.models.Word2Vec.load('brown.embedding')

MAX_LENGTH = 25  # Maximum sentence length to consider
MIN_COUNT = 2    # Minimum word count threshold for trimming

###########################################################
## Important: modify this to your directory of interest:
###########################################################
os.environ['MACHINE_LEARNING_DATA_HOME'] = 'A:/machine_learning/data'  # 'E:/machine_learning_udacity/data'

DATA_HOME = os.environ['MACHINE_LEARNING_DATA_HOME']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################
# modified from https://www.guru99.com/seq2seq-model.html
#########################
def indexes_from_sentence(voc: Voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')]


#########################
# modified from https://www.guru99.com/seq2seq-model.html
#########################
def tensor_from_sentence(voc: Voc, sentence):
    indexes = indexes_from_sentence(voc, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


#########################
# modified from https://www.guru99.com/seq2seq-model.html
#########################
def tensor_from_pair(voc, pair):
    input_tensor = tensor_from_sentence(voc, pair[0])
    target_tensor = tensor_from_sentence(voc, pair[1])
    return input_tensor, target_tensor


def load_dataframe(path, dataset_name):
    """
    You will use this function to load the dataset into a Pandas Dataframe for processing.
    """
    # torchtext.datasets.SQuAD1(root: str = '.data', split: Union[Tuple[str], str] = ('train', 'dev'))
    if dataset_name == 'squad1':
        train_iter, dev_iter = torchtext.datasets.SQuAD1(path)  #, split=('train', 'test'))
    elif dataset_name == 'squad2':
        train_iter, dev_iter = torchtext.datasets.SQuAD2(path)
    else:
        raise Exception('Dataset {} not found'.format(dataset_name))

    df_train = {
        "question": [],
        "answer": []
    }
    df_val = {
        'question': [],
        'answer': []
    }

    for context, question, answers, indices in train_iter:
        if answers[0]:
            df_train["question"].append(question)
            df_train["answer"].append(answers[0])

    for context, question, answers, indices in dev_iter:
        if answers[0]:
            df_val['question'].append(question)
            df_val['answer'].append(answers[0])

    return pd.DataFrame.from_dict(df_train), pd.DataFrame.from_dict(df_val)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
# Taken from LSTM Chatbot tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
# Taken from LSTM Chatbot tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
# Taken from LSTM Chatbot tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
def filter_pairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Taken from LSTM Chatbot tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
def trim_rare_words(voc, pairs, min_count):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# Modified from LSTM Chatbot tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
def create_vocab_object(df_train: pd.DataFrame, df_valid: pd.DataFrame, dataset_name: str):
    """
    :param df_train: Dataframe for all the question and answers
    :return: Voc object summarizing the vocabulary in the dataset
    """
    print("Reading lines...")
    # Read the file and split into lines
    # Split every line into pairs and normalize
    pairs_train = [[normalize_string(s) for s in l] for l in df_train.values]
    pairs_valid = [[normalize_string(s) for s in l] for l in df_valid.values]
    voc = Voc(dataset_name)

    print("Read {!s} sentence pairs (train)".format(len(pairs_train)))
    pairs_train = filter_pairs(pairs_train)
    print("Trimmed to {!s} sentence pairs (train)".format(len(pairs_train)))

    print('Read {!s} sentence pairs (valid)'.format(len(pairs_valid)))
    pairs_valid = filter_pairs(pairs_valid)
    print('Trimmed to {!s} sentence pairs (valid)'.format(len(pairs_valid)))

    print("Counting words...")
    for p_t, p_v in zip(pairs_train, pairs_valid):
        voc.addSentence(p_t[0])
        voc.addSentence(p_t[1])

        voc.addSentence(p_v[0])
        voc.addSentence(p_v[1])

    print("Counted words:", voc.num_words)
    return voc, pairs_train, pairs_valid


def get_vocab_and_sentence_pairs(dataset_name):

    df_train, df_valid = load_dataframe(DATA_HOME, dataset_name)
    voc, pairs_train, pairs_valid = create_vocab_object(df_train, df_valid, dataset_name)
    pairs_train = trim_rare_words(voc, pairs_train, MIN_COUNT)
    pairs_valid = trim_rare_words(voc, pairs_valid, MIN_COUNT)

    return voc, pairs_train, pairs_valid
