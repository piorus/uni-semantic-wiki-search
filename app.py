import glob
import json

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
WIKIEXTRACTOR_OUTPUT_DIR = 'text'


# pd.set_option('display.max_columns', None)

def wiki_pages_generator():
    for filename in glob.iglob(WIKIEXTRACTOR_OUTPUT_DIR + '/**/wiki_*', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                yield json.loads(line.rstrip())


def prepare_text(text):
    # return ".".join(text.split('\n')[0].split('.')[:2])
    return ".".join(text.split('\n'))


pages = {}
for page in wiki_pages_generator():
    pages[page['title']] = page

corpus = [prepare_text(page['text']) for page in wiki_pages_generator() if page['text'] != '']
index = [page['title'] for page in wiki_pages_generator() if page['text'] != '']

# N = 10
# corpus = corpus[-N:]
# index = index[-N:]

print('corpus = \n{}\n'.format('\n'.join(corpus)))

# nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['"', '(', ')', '.', ',', '–', 'us', 'u', 'cat', 'cats', 'behind', ';']
stop_words += list(map(str, list(range(0, 2022))))
stop_words += ['january', 'february', 'march', 'april', 'may', 'august', 'september', 'october', 'november']
# stop_words += ['is a after his he became with an for in'.split()]
# stop_words = list(sorted())
#
# vocab = []
# for v in index:
#     vocab += nltk.tokenize.casual.casual_tokenize(v.lower().replace('(', '').replace(')', ''), strip_handles=True)
#
# vocab = list(sorted(set(vocab)))
#
# for stop_word in stop_words:
#     if stop_word in vocab:
#         vocab.remove(stop_word)
#
# print('vocab = ', vocab)


def search(Q, index):
    # Q = ['what are the name of cats of president of taiwan?']
    # Q = ['nyan cat']

    query = nltk.tokenize.casual.casual_tokenize(Q)
    query = list(sorted(set(query)))

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=nltk.tokenize.casual.casual_tokenize,
        stop_words=stop_words,
        vocabulary=query,
        min_df=0.1,
        max_df=.99
    )
    tfidf_docs = pd.DataFrame(tfidf_vectorizer.fit_transform(raw_documents=corpus).todense(), index=index)
    id_words = [(i, w) for (w, i) in tfidf_vectorizer.vocabulary_.items()]
    tfidf_docs.columns = list(zip(*sorted(id_words)))[1]

    print('tfidf_docs = \n', tfidf_docs)


    pca = PCA(n_components=1)
    pca.fit(tfidf_docs.values)
    pca_topic_vectors = pca.transform(tfidf_docs.values)

    pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=[Q], index=index)
    print('\npca_topic_vectors = \n', pca_topic_vectors)

    tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=[' '.join(query)]).toarray())
    pca_q = pca.transform(tfidf_q.values)
    print('\npca_q = ', pca_q, '\n')

    # distances = []
    min_distance = 1
    min_topic = None

    for index, row in pca_topic_vectors.iterrows():
        distance = abs(row[Q] - pca_q[0][0])

        print(
            'topic = "{}", '.format(index),
            'avg = {}, '.format(row[Q]),
            'pca_q = {}, '.format(pca_q[0][0]),
            'distance = {}, '.format(distance)
        )

        if distance < min_distance:
            min_distance = distance
            min_topic = index

    print('\nmin_distance = ', min_distance)
    print('min_topic = ', min_topic)

while True:
    search(input('> '), index)