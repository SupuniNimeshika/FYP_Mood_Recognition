import re
import string
import nltk
import pandas as pd

fullCorpus = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
fullCorpus.columns = ['label', 'body_text']
fullCorpus.head()

print("input data has {} rows columns".format(len(fullCorpus), len(fullCorpus.columns)))

print("Out of {} rows , {} are spam , {} are ham".format(len(fullCorpus),
                                                         len(fullCorpus[fullCorpus['label'] == 'spam']),
                                                         len(fullCorpus[fullCorpus['label'] == 'ham'])))

print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
print("Number of null in label: {}".format(fullCorpus['body_text'].isnull().sum()))
