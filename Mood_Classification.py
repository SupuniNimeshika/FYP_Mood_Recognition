import re
import string
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

fullCorpus = pd.read_csv('new.tsv', sep='|', header=None)
fullCorpus.columns = ['label', 'body_text']
fullCorpus.head()

print("input data has {} rows columns".format(len(fullCorpus), len(fullCorpus.columns)))

print("Out of {} rows , {} are happy , {} are sad , {} are angry , {} are calm".format(len(fullCorpus),
                                                         len(fullCorpus[fullCorpus['label'] == 'happy']),
                                                         len(fullCorpus[fullCorpus['label'] == 'sad']),
                                                         len(fullCorpus[fullCorpus['label'] == 'angry']),
                                                         len(fullCorpus[fullCorpus['label'] == 'calm'])))

print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
print("Number of null in label: {}".format(fullCorpus['body_text'].isnull().sum()))


# remove punctuation
string.punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct
fullCorpus['body_text_clean'] = fullCorpus['body_text'].apply(lambda x: remove_punct(x))
fullCorpus.head(25)

# print seperate file for remove punctuation output
print('----------------------------Print in the punctuation.txt-----------------------------------------')
file = open('punctuation.txt', 'w')
punctuationed = fullCorpus['body_text_clean']
for index, val in punctuationed.iteritems():
    line = str(index)+'\t'+str(val)
    file.write(line+'\n')
file.close()


# tokenization
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens
fullCorpus['body_text_tokenized'] = fullCorpus['body_text_clean'].apply(lambda x: tokenize(x.lower()))
fullCorpus.head()

# print seperate file for tokenize output
print('----------------------------Print in the tokenize.txt-----------------------------------------')
file = open('tokenize.txt', 'w')
tokenized = fullCorpus['body_text_tokenized']
for index, val in tokenized.iteritems():
    line = str(index)+'\t'+str(val)
    file.write(line+'\n')
file.close()


# stop word
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text
fullCorpus['body_text_nostop'] = fullCorpus['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
fullCorpus.head()

# print seperate file for remove stop word output
print('----------------------------Print in the nonstop.txt-----------------------------------------')
file = open('nostop.txt', 'w')
nostoped = fullCorpus['body_text_nostop']
for index, val in nostoped.iteritems():
    line = str(index)+'\t'+str(val)
    file.write(line+'\n')
file.close()


#lemmatizing
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

# print seperate file for lematization output
print('----------------------------Print in the cleaned.txt-----------------------------------------')
file = open('cleaned.txt', 'w')
cleaned=fullCorpus['body_text_lemmatized']
for index, val in cleaned.iteritems():
    line = str(index)+'\t'+val
    file.write(line+'\n')
file.close()


# separate training and testing data
trainData = fullCorpus['body_text_lemmatized'][:350]
testData = fullCorpus['body_text_lemmatized'][350:]
train_labels = fullCorpus['label'][:350]
test_labels = fullCorpus['label'][350:]

# print train data and test data
print("Train data count:\n", train_labels.value_counts())
print("Test data count:\n", test_labels.value_counts())

#extract Unigrams
unigram_vectorized = CountVectorizer(stop_words="english",analyzer="word",ngram_range=(1,1), max_df=1.0,min_df=1,max_features=None)
count_unigram = unigram_vectorized.fit(trainData)
unigrams = unigram_vectorized.transform(trainData)

#extract Bigrams
bigram_vectorized = CountVectorizer(stop_words="english",analyzer="word",ngram_range=(2,2), max_df=1.0,min_df=1,max_features=None)
count_bigram = bigram_vectorized.fit(trainData)
bigrams = bigram_vectorized.transform(trainData)

#extract Trigrams
trigram_vectorized = CountVectorizer(stop_words="english",analyzer="word",ngram_range=(3,3), max_df=1.0,min_df=1,max_features=None)
count_trigram = trigram_vectorized.fit(trainData)
trigrams = trigram_vectorized.transform(trainData)

# unigram,bigram and trigram as together
full_vectorized = CountVectorizer(stop_words="english",analyzer="word",ngram_range=(1,3), max_df=1.0,min_df=1,max_features=None)
count_full = full_vectorized.fit(trainData)
full = full_vectorized.transform(trainData)

# unigram frequency
unigram_fre = TfidfTransformer().fit(unigrams)
transformer_unigrams = unigram_fre.transform(unigrams)

#Bigram frequency
bigram_fre = TfidfTransformer().fit(bigrams)
transformer_bigrams = bigram_fre.transform(bigrams)

#Trigram frequency
trigram_fre = TfidfTransformer().fit(trigrams)
transformer_trigrams = trigram_fre.transform(trigrams)

#full set frequency
full_fre = TfidfTransformer().fit(full)
transformer_full = full_fre.transform(full)

# define model for n-grams
u_model = MultinomialNB().fit(transformer_unigrams,train_labels)
b_model = MultinomialNB().fit(transformer_bigrams,train_labels)
t_model = MultinomialNB().fit(transformer_trigrams,train_labels)
f_model = MultinomialNB().fit(transformer_full,train_labels)


# sample for testing comment
test_sample = ['Missing you DAD','After reading the story tears came out of my eyes']

#Unigram probability counting
uni_test_CountVectorized = unigram_vectorized.transform(test_sample)
uni_test_fre = unigram_fre.transform(uni_test_CountVectorized)
unimodel_test_CountVectorized = unigram_vectorized.transform(testData)
unimodel_test_fre = unigram_fre.transform(unimodel_test_CountVectorized)

#Bigram probability counting
bi_test_CountVectorized = bigram_vectorized.transform(test_sample)
bi_test_fre = bigram_fre.transform(bi_test_CountVectorized)
bimodel_test_CountVectorized = bigram_vectorized.transform(testData)
bimodel_test_fre = bigram_fre.transform(bimodel_test_CountVectorized)

#Trigram probability counting
tri_test_CountVectorized = trigram_vectorized.transform(test_sample)
tri_test_fre = trigram_fre.transform(tri_test_CountVectorized)
trimodel_test_CountVectorized = trigram_vectorized.transform(testData)
trimodel_test_fre = trigram_fre.transform(trimodel_test_CountVectorized)

#Full set probability counting
full_test_CountVectorized = full_vectorized.transform(test_sample)
full_test_fre = full_fre.transform(full_test_CountVectorized)
full_test_CountVectorized = full_vectorized.transform(testData)
full_test_fre = full_fre.transform(full_test_CountVectorized)


# output final results
print("\n        Unigram Model result:")
unigram_result = u_model.predict(uni_test_fre)
print(unigram_result)
unigrammodel_result = u_model.predict(unimodel_test_fre)
print (classification_report(test_labels, unigrammodel_result))

print("\n            Bigram Model result:")
bigram_result = b_model.predict(bi_test_fre)
print(bigram_result)
bigrammodel_result = b_model.predict(bimodel_test_fre)
print (classification_report(test_labels, bigrammodel_result))

print("\n            Trigram Model result:")
trigram_result = t_model.predict(tri_test_fre)
print(trigram_result)
trigrammodel_result = t_model.predict(trimodel_test_fre)
print (classification_report(test_labels, trigrammodel_result))

print("\n            Full Model result:")
full_set_result = f_model.predict(full_test_fre)
print(full_set_result)
full_model_result = f_model.predict(full_test_fre)
print (classification_report(test_labels, full_model_result))


