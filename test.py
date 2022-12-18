new_york = """New York, often called New York City to distinguish it from New York State, or NYC for short, is the most populous city in the United States. With a 2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), New York City is also the most densely populated major city in the United States. Located at the southern tip of the State of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban area. With over 20 million people in its metropolitan statistical area and approximately 23 million in its combined statistical area, it is one of the world's most populous megacities. New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce, entertainment, research, technology, education, politics, tourism, dining, art, fashion, and sports, and is the most photographed city in the world. Home to the headquarters of the United Nations, New York is an important center for international diplomacy, and has sometimes been called the capital of the world."""

york = """York is the oldest inland town in Western Australia, situated on the Avon River, 97 kilometres (60 mi) east of Perth in the Wheatbelt, on Ballardong Nyoongar land, and is the seat of the Shire of York. The name of the region was suggested by JS Clarkson during an expedition in October 1830 because of its similarity to his own county in England, Yorkshire. After thousands of years of occupation by Ballardong Nyoongar people, the area was first settled by Europeans in 1831, two years after Perth was settled in 1829. A town was established in 1835 with the release of town allotments and the first buildings were erected in 1836. The region was important throughout the 19th century for sheep and grain farming, sandalwood, cattle, goats, pigs and horse breeding. York boomed during the gold rush as it was one of the last rail stops before the walk to the goldfields. Today, the town attracts tourists for its beauty, history, buildings, festivals and art. """

import glob, json
def wiki_pages_generator():
    for filename in glob.iglob("text" + '/**/wiki_*', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                yield json.loads(line.rstrip())


def prepare_text(text):
    sentences = text.split('\n')[0].split('.')
    sentences = sentences[:2]
    return ".".join(sentences)


corpus = [prepare_text(page['text']) for page in wiki_pages_generator() if page['text'] != '']
corpus = corpus[-3:]
print('corpus = \n{}\n'.format('\n'.join(corpus)))
index = ['Nyan Cat', 'Grumpy Cat', 'Hank the Cat']
# corpus = [new_york, york]
# index = ['new york', 'york']

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
# 0.22688429
# 0.21293172
tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=Q).toarray())
pca_q = pca.transform(tfidf_q)
print('pca_q = ', pca_q)