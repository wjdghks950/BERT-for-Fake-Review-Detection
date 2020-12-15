import os
import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils import get_label, MODEL_CLASSES
from data_loader import tokenize_review

logger = logging.getLogger(__name__)


class GoldPredictor(object):
    def __init__(self):
        self.model_type = 'bert'
        self.model_name_or_path = 'bert-base-uncased'
        self.task = 'opspam'
        self.model_dir = './model'
        self.max_seq_len = 512

        self.label_lst = [0, 1]
        self.num_labels = len(self.label_lst)

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_type]

        self.config = self.config_class.from_pretrained(self.model_name_or_path,
                                                        num_labels=self.num_labels, 
                                                        finetuning_task=self.task, output_attentions=True)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name_or_path)
        self.model = self.model_class.from_pretrained(self.model_dir, config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        print("*****************Config & Pretrained Model load complete**********************")

    def predict(self, review):
        tokenized_review = tokenize_review(self.max_seq_len, self.tokenizer, self.device, review)

        with torch.no_grad():
            inputs = {'input_ids': tokenized_review[0],
                        'attention_mask': tokenized_review[1],
                        'token_type_ids': tokenized_review[2]}
            outputs = self.model(**inputs)
            normalized_scores = F.softmax(outputs[0], dim=-1).detach().cpu()[0].tolist()
            prediction = np.argmax(normalized_scores).item()
            prediction = 'Gold' if prediction == 0 else 'Fake'
            review_words, cls_attention_words = self.get_cls_attentions(review, outputs[1])

        return prediction, normalized_scores, review_words, cls_attention_words

    def get_cls_attentions(self, review, raw_attentions):
        last_attention_scores = raw_attentions[-1]
        head_avg_attention_scores = last_attention_scores.squeeze(0).mean(dim=0)
        cls_attentions = head_avg_attention_scores[0]

        review_words = review.lower().split()
        review_tokens = ['[CLS]'] + self.tokenizer.tokenize(review) + ['[SEP]']

        cls_attentions = cls_attentions.detach().cpu().tolist()
        while 0.0 in cls_attentions:    
            cls_attentions.remove(0.0)

        max_attention_list_len = len(cls_attentions) if len(cls_attentions) < self.max_seq_len else self.max_seq_len
        review_tokens = review_tokens[1:max_attention_list_len-1]
        cls_attentions_tokens = cls_attentions[1:max_attention_list_len-1]
        
        cls_attentions_words = []
        for word in review_words:
            for i in range(len(review_tokens)):
                token = ''.join(review_tokens[:i+1]).replace('#', '')
                if word == token:
                    tokens_mean = np.mean(cls_attentions_tokens[:i+1])
                    cls_attentions_words.append(tokens_mean)
                    del cls_attentions_tokens[:i+1]
                    del review_tokens[:i+1]
                    break

        return review_words[:len(cls_attentions_words)], cls_attentions_words
        
model = GoldPredictor()

def get_model():
    return model


# if __name__ == '__main__':

#     model = GoldPredictor()

#     while True:
#         review = input('input: ')
#         outputs = model.predict(review)
#         print(outputs)