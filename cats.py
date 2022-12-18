corpus = [
    "Hank the Cat (August 2001 – February 13, 2014) was a Maine Coon cat that was put up as a joke candidate in the 2012 United States Senate election in Virginia, a feat which gained international coverage after Hank reportedly came third behind the two major candidates. He died in 2014 due to stomach lymphoma",
    "Nyan Cat is a YouTube video uploaded in April 2011, which became an internet meme. The video merged a Japanese pop song with an animated cartoon cat with a Pop-Tart for a torso flying through space and leaving a rainbow trail behind",
    "Tardar Sauce (April 4, 2012 – May 14, 2019), nicknamed Grumpy Cat, was an American Internet celebrity cat. She was known for her permanently \"grumpy\" facial appearance, which was caused by an underbite and feline dwarfism"
]
index = ['Hank the Cat', 'Nyan Cat', 'Grumpy Cat']

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')

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
tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=Q).toarray())
pca_q = pca.transform(tfidf_q)
print('pca_q = ', pca_q)

min_distance = 1
min_topic = None

for index, row in pca_topic_vectors.iterrows():
    distance = abs(row['topic0'] - pca_q[0][0])
    # distance = abs(pca_q - row['topic0'])[0][0]
    print('topic = "{}", '.format(index), 'pca_vec = {}, '.format(row['topic0']), 'pca_q = {}, '.format(pca_q[0][0]),
          'distance = {}, '.format(distance))

    if distance < min_distance:
        min_distance = distance
        min_topic = index

print('min_distance = ', min_distance)
print('min_topic = ', min_topic)