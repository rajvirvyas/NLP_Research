from sklearn.feature_extraction.text import CountVectorizer
import pandas


df = pandas.read_excel('replies.xlsx',sheet_name='Sheet1', usecols='A')
df.fillna(' ')
df = df.reset_index()
df.fillna(' ')

data = []

c_vec = CountVectorizer(ngram_range=(1, 4))

ngrams = c_vec.fit_transform(df.iloc[:, 1].values.astype('U'))

vocab = c_vec.vocabulary_

count_values = ngrams.toarray().sum(axis=0)

for ng_count, ng_text in sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True):
    print(ng_count, ng_text)