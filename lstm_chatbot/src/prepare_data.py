import gensim
import nltk
import numpy as np
import pandas as pd
import gzip
import torch
import torchtext
from nltk.corpus import brown

nltk.download('brown')
nltk.download('punkt')

# Output, save, and load brown embeddings

model = gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding')

w2v = gensim.models.Word2Vec.load('brown.embedding')


def loadDF(path):
    """
    You will use this function to load the dataset into a Pandas Dataframe for processing.
    """
    # torchtext.datasets.SQuAD1(root: str = '.data', split: Union[Tuple[str], str] = ('train', 'dev'))
    train_iter, dev_iter = torchtext.datasets.SQuAD1(path)  #, split=('train', 'test'))

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


if __name__ == '__main__':
    f = loadDF('./')
    print('yo')
