import glob
import json

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

WIKIEXTRACTOR_OUTPUT_DIR = 'text'


def prepare_text(text):
    return text


def wiki_pages_generator():
    for filename in glob.iglob(WIKIEXTRACTOR_OUTPUT_DIR + '/**/wiki_*', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                yield json.loads(line.rstrip())


def get_vectorizer(stop_words: list = None, vocabulary: list = None):
    return TfidfVectorizer(
        tokenizer=nltk.tokenize.casual.casual_tokenize,
        stop_words=stop_words,
        vocabulary=vocabulary,
    )


def build_tfidf_docs(tfidf_vectorizer: TfidfVectorizer, corpus: list, index: list):
    tfidf_docs = pd.DataFrame(tfidf_vectorizer.fit_transform(raw_documents=corpus).todense(), index=index)
    id_words = [(i, w) for (w, i) in tfidf_vectorizer.vocabulary_.items()]
    tfidf_docs.columns = list(zip(*sorted(id_words)))[1]

    print('tfidf_docs = \n', tfidf_docs)

    return tfidf_docs


def search(Q: str, tfidf_vectorizer: TfidfVectorizer, tfidf_docs: pd.DataFrame, index: list):
    pca = PCA(n_components=1)
    pca.fit(tfidf_docs.values)
    pca_topic_vectors = pca.transform(tfidf_docs.values)
    pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=[Q], index=index)
    print('\npca_topic_vectors = \n', pca_topic_vectors)

    tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=[Q]).toarray())
    pca_q = pca.transform(tfidf_q.values)
    print('\npca_q = ', pca_q, '\n')

    min_distance = 1
    min_topic = None

    for page_title, row in pca_topic_vectors.iterrows():
        distance = abs(row[Q] - pca_q[0][0])

        print(
            'topic = "{}", '.format(page_title),
            'avg = {}, '.format(row[Q]),
            'pca_q = {}, '.format(pca_q[0][0]),
            'distance = {}, '.format(distance)
        )

        if distance < min_distance:
            min_distance = distance
            min_topic = page_title

    print('\nmin_distance = ', min_distance)
    print('min_topic = ', min_topic)

    return min_topic


pages = {page['title']: page for page in wiki_pages_generator() if page['text'] != ''}
corpus = [prepare_text(page['text']) for page in pages.values()]
index = [page['title'] for page in pages.values()]
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['known']

while True:
    Q = input('> ')
    query = nltk.tokenize.casual.casual_tokenize(Q)
    query = list(sorted(set(query)))

    tfidf_vectorizer = get_vectorizer(stop_words, vocabulary=query)
    tfidf_docs = build_tfidf_docs(tfidf_vectorizer, corpus=corpus, index=index)
    found_page_title = search(Q, tfidf_vectorizer, tfidf_docs=tfidf_docs, index=index)
    print('\n=============================')
    print('Found page: ', found_page_title)
    print('Page content: ', pages.get(found_page_title)['text'])
