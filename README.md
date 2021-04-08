# Fake Review Detection for restaurants, products and hotels

With a growing interest in the field of anomaly detection came along the tasks for ``Fake Review Detection``.
Fake Review Detection is a task for detecting fake, anomalous reviews among a given text corpus of reviews.
In this work, with the ``Opspam dataset`` (and ``Amazon dataset``), we provide a BERT-based fake review detection model that achieves an accuracy of 0.8991 (``Acc = 0.8991``).

- Fake review detection using BERT (`bert-base-uncased`)
- Using `Huggingface Tranformers` library to implement the fake review detection model
- Dataset: Opspam dataset, Amazon fake review dataset, Yelp dataset

## Dependencies

- torch==1.4.0
- transformers==2.10.0

Install them with the following command:

$ pip install -r requirements.txt

## Usage

Training with Opspam dataset:
```bash
$ ./train_opspam.sh
```

Training with Amazon dataset:
```bash
$ ./train_amazon.sh
```

Or, you can simply unzip the `data.zip` to train and get going with model evaluation!

$ unzip data.zip -d ./data

## References
References for dataset and the Huggingface Transformers library
- [Deceptive Opinion Spam Corpus](https://myleott.com/op-spam.html)
- [Amazon Fake Review Dataset](https://www.kaggle.com/lievgarcia/amazon-reviews)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
