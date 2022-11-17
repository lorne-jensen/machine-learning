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

from src.data_to_tensors import batch_to_train_data
from src.vocab import Voc, EOS_token

nltk.download('brown')
nltk.download('punkt')

# Output, save, and load brown embeddings

model = gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding')

w2v = gensim.models.Word2Vec.load('brown.embedding')

MAX_LENGTH = 15  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming

DATA_HOME = 'A:/machine_learning/data'
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
    else:
        raise Exception('Dataset {} not found'.format(dataset_name))

    df = {
        "question": [],
        "answer": []
    }

    for context, question, answers, indices in train_iter:
        if answers[0]:
            df["question"].append(question)
            df["answer"].append(answers[0])

    return pd.DataFrame.from_dict(df)


def prepare_text(sentence):
    """
    Our text needs to be cleaned with a tokenizer. This function will perform that task.
    https://www.nltk.org/api/nltk.tokenize.html
    """
    tokens = nltk.tokenize.sent_tokenize(sentence)
    # tokens = [nltk.tokenize.word_tokenize(t) for t in tokens]  # TODO not sure yet...
    return tokens


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
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
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
def create_vocab_object(df: pd.DataFrame, dataset_name: str):
    """
    :param df: Dataframe for all the question and answers
    :return: Voc object summarizing the vocabulary in the dataset
    """
    print("Reading lines...")
    # Read the file and split into lines
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l] for l in df.values]
    voc = Voc(dataset_name)

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def train_test_split(SRC, TRG):
    """
    Input: SRC, our list of questions from the dataset, should be a tuple of (train_df, test_df)
            TRG, our list of responses from the dataset

    Output: Training and test datasets for SRC & TRG

    """
    SRC_train_dataset = SRC[0]
    SRC_test_dataset = SRC[1]
    TRG_train_dataset = TRG[0]
    TRG_test_dataset = TRG[1]
    # SRC_train_dataset = train_df["question"].tolist()
    # TRG_train_dataset = train_df["answer"].tolist()
    #
    # SRC_test_dataset = test_df["question"].tolist()
    # TRG_test_dataset = test_df["answer"].tolist()
    return SRC_train_dataset, SRC_test_dataset, TRG_train_dataset, TRG_test_dataset


def get_batches_from_dataset(dataset_name, batch_length):

    df = load_dataframe(DATA_HOME, dataset_name)
    voc, pairs = create_vocab_object(df, dataset_name)
    pairs = trim_rare_words(voc, pairs, MIN_COUNT)

    batches = batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_length)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    return input_variable, lengths, target_variable, mask, max_target_len, voc, pairs


if __name__ == '__main__':
    get_batches_from_dataset('squad1', 5)
