import os
import copy
import logging
import pandas as pd
import csv
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class OpSpamExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        deceptive: Indidcator for fake or not (`deceptive` vs. truthful) : Binary
        hotel: name of the hotel
        polarity: negative or positive polarity
        source: Review website
        text: Review text
    """

    def __init__(self, deceptive, hotel, polarity, source, text):
        self.deceptive = deceptive
        self.hotel = hotel
        self.polarity = polarity
        self.source = source
        self.text = text

    def __repr__(self):
        fake = "Yes" if self.deceptive == 1 else "No"
        delineate = "#" * 50
        s = "\nFake: {}\nHotel: {}\nPolarity: {}\nSource: {}\nReview Text: {}\n".format(fake, self.hotel, self.polarity, self.source, self.text)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class OpSpamFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        fake = "Yes" if self.deceptive == "deceptive" else "No"
        delineate = "#" * 50
        s = "\nFake: {}\nHotel: {}\nPolarity: {}\nSource: {}\nReview Text: {}\n".format(fake, self.hotel, self.polarity, self.source, self.text)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class OpSpamProcessor(object):
    """Processor for the Deceptive Opinion Spam Prediction data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """ Read the Deceptive Opinion Spam File"""
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df = df[df['text'].notna()]
        else:
            raise Exception("Error: {} does not exist".format(input_file))
        return df

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, row in df.iterrows():
            deceptive = row["deceptive"]
            hotel = row["hotel"]
            polarity = row["polarity"]
            source = row["source"]
            text = row["text"]
            # print("Text >> {}".format(text))
            # print("Label >> {} ".format(deceptive))
            if i % 500 == 0:
                logger.info(row)
            examples.append(OpSpamExample(deceptive=deceptive, hotel=hotel, polarity=polarity, source=source, text=text))

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


class YelpProcessor(object):
    """Processor for the Yelp review data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        pass

    def _create_examples(self, df, set_type):
        pass

    def get_examples(self, mode):
        pass


class AmazonExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        DOC_ID: Document ID
        LABEL: `__label1__` : Fake
        RATING: Rating per review
        VERIFIED_PURCHASE: Y / N
        PRODUCT_CATEGORY: PC, BABY, etc.
        PRODUCT_ID: Unique 10 alphanumeric code for product ID
        PRODUCT_TITLE: Product name
        REVIEW_TITLE: Review title
        REVIEW_TEXT: Review text
    """

    def __init__(self, doc_id, label, rating, verified, prod_category, prod_id, prod_title, rev_title, rev_text):
        self.doc_id = doc_id
        self.label = label
        self.rating = rating
        self.verified = verified
        self.prod_category = prod_category
        self.prod_id = prod_id
        self.prod_title = prod_title
        self.rev_title = rev_title
        self.text = rev_text

    def __repr__(self):
        fake = "Yes" if self.label == "__label1__" else "No"
        delineate = "#" * 50
        s = "\nFake: {}\nProduct: {}\nReview Title: {}\nReview Text: {}\n".format(fake, self.prod_title, self.rev_title, self.text)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class AmazonFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        fake = "Yes" if self.label == "__label1__" else "No"
        delineate = "#" * 50
        s = "\nFake: {}\nProduct: {}\nReview Title: {}\nReview Text: {}\n".format(fake, self.prod_title, self.rev_title, self.text)
        s = delineate + s + delineate
        return s

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class AmazonProcessor(object):
    """Processor for the Amazon review data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """ Read the Deceptive Opinion Spam File"""
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            # df = df[df['text'].notna()]
        else:
            raise Exception("Error: {} does not exist".format(input_file))
        return df

    def _create_examples(self, df, set_type):
        """Creates examples for the Amazon training and dev sets."""
        examples = []
        for i, row in df.iterrows():
            
            doc_id = row["DOC_ID"]
            label = row["LABEL"]
            rating = row["RATING"]
            verified = row["VERIFIED_PURCHASE"]
            prod_category = row["PRODUCT_CATEGORY"]
            prod_id = row["PRODUCT_ID"]
            prod_title = row["PRODUCT_TITLE"]
            rev_title = row["REVIEW_TITLE"]
            rev_text = row["REVIEW_TEXT"]
            if i % 500 == 0:
                logger.info(row)
            examples.append(AmazonExample(doc_id=doc_id, label=label, rating=rating,
                                          verified=verified, prod_category=prod_category,
                                          prod_id=prod_id, prod_title=prod_title,
                                          rev_title=rev_title, rev_text=rev_text))

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "opspam": OpSpamProcessor,
    "yelp": YelpProcessor,
    "amazon": AmazonProcessor
}


def convert_examples_to_features(args, examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer.tokenize(example.text)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = 0
        if args.task == "opspam":
            if example.deceptive == "deceptive":
                label_id = 1
            elif example.deceptive == "truthful":
                label_id = 0
            else:
                continue  # Skip this data instance
        elif args.task == "amazon":
            if example.label == "__label1__":  # `__label1__ == fake`
                label_id = 1
            else:
                label_id = 0

        if args.task == "opspam":
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("Hotel: %s" % example.hotel)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d) (Fake == 1 / Real == 0)" % (example.deceptive, label_id))
            features.append(
                OpSpamFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=label_id
                    ))

        elif args.task == "amazon":
            features.append(
                AmazonFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=label_id
                    ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(args, examples, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    print("*********Dataset preprocessing complete**********")
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    return dataset
