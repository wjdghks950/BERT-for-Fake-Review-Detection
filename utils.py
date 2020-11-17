import random
import logging

import torch
import numpy as np

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    BertTokenizer,
    ElectraTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForSequenceClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-uncased',  # 'bert-base-multilingual-cased'
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }


def precision(pred, labels):
    # tp / (tp + fp)
    tp = sum((pred == 1) & (labels == 1))
    fp = sum((pred == 1) & (labels == 0))
    return tp / (tp + fp)


def recall(pred, labels):
    # tp / (tp + fn)
    tp = sum((pred == 1) & (labels == 1))
    fn = sum((pred == 0) & (labels == 1))
    return tp / (tp + fn)


def f1_score(pred, labels):
    prec = precision(pred, labels)
    rec = recall(pred, labels)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("Training process is stopped early...")
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
