# Fake Review Detection for restaurants, products and hotels

- Fake review detection using BERT (`bert-base-uncased`)
- Using `Huggingface Tranformers` library to implement the fake review detection model
- Dataset: Opspam dataset, Amazon fake review dataset, Yelp dataset

## Dependencies

- torch==1.4.0
- transformers==2.10.0

## Usage

```bash
$ python main.py --task opspam --do_train
```

## References
References for dataset and the Huggingface Transformers library
- [Deceptive Opinion Spam Corpus](https://myleott.com/op-spam.html)
- [Amazon Fake Review Dataset](https://www.kaggle.com/lievgarcia/amazon-reviews)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
