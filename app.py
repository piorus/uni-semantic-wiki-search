import glob
import json

WIKIEXTRACTOR_OUTPUT_DIR = 'text'


def wiki_pages_generator():
    for filename in glob.iglob(WIKIEXTRACTOR_OUTPUT_DIR + '/**/wiki_*', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                yield json.loads(line.rstrip())


def prepare_text(text):
    return text.split('\n')[0]


corpus = [prepare_text(page['text']) for page in wiki_pages_generator() if page['text'] != '']
index = [page['title'] for page in wiki_pages_generator() if page['text'] != '']

N = 5

corpus = corpus[-N:]
index = index[-N:]

print('corpus = \n{}\n'.format('\n'.join(corpus)))

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['(', ')', '#', '-', ',', '"']

tfidf_vectorizer = TfidfVectorizer(
    tokenizer=nltk.tokenize.casual.casual_tokenize,
    stop_words=stop_words
)
tfidf_docs = pd.DataFrame(tfidf_vectorizer.fit_transform(raw_documents=corpus).todense(), index=index)
id_words = [(i, w) for (w, i) in tfidf_vectorizer.vocabulary_.items()]
tfidf_docs.columns = list(zip(*sorted(id_words)))[1]

print('tfidf_docs = \n', tfidf_docs)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print('pca_topic_vectors = \n', pca_topic_vectors)

Q = ['nyan cat']
# 0.22688429
# 0.21293172
tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=Q).toarray())
pca_q = pca.transform(tfidf_q)
print('pca_q = ', pca_q)

# distances = []
min_distance = 1
min_topic = None


# for index, row in enumerate(pca_topic_vectors.iterrows()):
#     distance = abs(row[1]['topic{}'.format(index)] - pca_q[0][index])
#     print(distance)
    # for dimension in pca_q[0]:
    #     distance = abs(row['topic0'] - dimension)
    # # distance = abs(pca_q - row['topic0'])[0][0]
    # print('topic = "{}", '.format(index), 'pca_vec = {}, '.format(row['topic0']), 'pca_q = {}, '.format(pca_q[0][0]),
    #       'distance = {}, '.format(distance))
    #
    # if distance < min_distance:
    #     min_distance = distance
    #     min_topic = index

print('\n')
print('min_distance = ', min_distance)
print('min_topic = ', min_topic)
