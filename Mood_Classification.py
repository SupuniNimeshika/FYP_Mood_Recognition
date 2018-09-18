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


# # Read Csv File
# fullCorpus = pd.read_csv('HappySadDataSet.tsv', sep='\t', header=None)
# fullCorpus.columns = ['label', 'body_text']
# fullCorpus.head()
#
# print("input data has {} rows columns".format(len(fullCorpus), len(fullCorpus.columns)))
#
# print("Out of {} rows , {} are happy , {} are sad".format(len(fullCorpus),
#                                                          len(fullCorpus[fullCorpus['label'] == 'happy']),
#                                                          len(fullCorpus[fullCorpus['label'] == 'sad'])))
#
# print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
# print("Number of null in label: {}".format(fullCorpus['body_text'].isnull().sum()))



# tokenization
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens
fullCorpus['body_text_tokenized'] = fullCorpus['body_text'].apply(lambda x: tokenize(x.lower()))
fullCorpus.head()

# stop word
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text
fullCorpus['body_text_nostop'] = fullCorpus['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
fullCorpus.head()

# lemmatizing
wn = nltk.WordNetLemmatizer()

def clean_text(tokeized_text):
    text = [wn.lemmatize(word) for word in tokeized_text]
    return text

def join_text(sentence):
    return ' '.join(sentence)

def data_lematization():
    fullCorpus['body_text_lemmatized'] = fullCorpus['body_text_nostop'].apply(lambda x: clean_text(x))
    fullCorpus['body_text_lemmatized'] = fullCorpus['body_text_lemmatized'].apply(lambda x: join_text(x))
data_lematization()
