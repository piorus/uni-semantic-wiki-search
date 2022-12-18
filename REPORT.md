# Project Report

## Title

Semantic Wikipedia Search in Python

## Authors

Piotr Rusin

## Abstract

## 1. Introduction

### 1.1 Problem to solve

Construct a semantic search engine with a database based on Wikipedia (selected topic). Please use different methods to construct word vectors. Use Python libraries.

### 1.2 Selection of topic

As a topic, I selected cats because Wikipedia stores information about a wide range of species, and related pages are frequently updated by cat lovers. 

### 1.3 Wikipedia page export

Wikipedia allows anyone to download its database, and - as it turned out - this download can also be constrained to chosen pages.

To make the whole process less of a burden, Wikipedia founders created a tool called [Export pages](https://en.wikipedia.org/wiki/Special%3aExport) 
which allows you to select a category you wish to export, and then it automatically pulls a current list of pages that are within this category. 

I decided to use those pages as a corpus:
```
Creme_Puff_(cat)
Tuxedo_Stan
Think_Think_and_Ah_Tsai
Catmando
Freya_(cat)
Gladstone_(cat)
India_(cat)
Hank_the_Cat
Nyan_Cat
Grumpy_Cat
```

The unprocessed XML file that was exported as a result is located within this repository, [here](Wikipedia.xml).

## 2. Development

### 2.1 Parsing raw XML into JSON

Wikipedia uses XMLs for its dumps. Fortunately, there is a Python CLI tool called [wikiextractor](https://github.com/attardi/wikiextractor) that can smoothly convert it into JSON.


First, I installed it through pipenv:
```
pipenv install wikiextractor
```

Then, I converted the XML dump to JSON by executing:
```
pipenv run wikiextractor Wikipedia.xml --json
```

As a result, a new `text` directory was created within the project root with two files `wiki_00`, and `wiki_01`:

![wikiextractor1.png](report%2Fwikiextractor1.png)

with the following content:

![wikiextractor2.png](report%2Fwikiextractor2.png)

### 2.2 Parsing wikiextractor output

wikiextractor saved the page's JSON in separate files (useful for bigger exports). 

This output still needs to be loaded in Python, and the JSON string needs to be parsed into Python dictionary.

I decided to create a generator that recursively iterates over wikiextractor output files, parse each line as JSON, and yields it as a result:
```python
import glob
import json

WIKIEXTRACTOR_OUTPUT_DIR = 'text'

def wiki_pages_generator():
    for filename in glob.iglob(WIKIEXTRACTOR_OUTPUT_DIR + '/**/wiki_*', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                yield json.loads(line.rstrip())
```

I used this generator to prepare a corpus for the semantic search.

### 2.3 wikiextractor's JSON schema

```json
{
    "id": "4706881", 
    "revid": "21501829", 
    "url": "https://en.wikipedia.org/wiki?curid=4706881", 
    "title": "Page title", 
    "text": "Page content (plaintext)..."
}
```

### 2.4 Building a corpus

Just a simple dictionary and list comprehensions that load JSON objects from wikiextractor output, and append it into the corpus + index list:
```python
pages = {page['title']: page for page in wiki_pages_generator() if page['text'] != ''}
corpus = [page['text'] for page in pages.values()]
index = [page['title'] for page in pages.values()]
```

I decided to use page titles as an index for `pandas` printouts.

### 2.5 Stop words

Same, old `nltk.corpus.stopwords`:

```
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['known']
```

I've appended a list of additional stop words, but as of now only `known` landed there.

### 2.6 REPL

Loop consists of logic that is responsible for:
* awaiting for user input - needed to process semantic search over corpus
* user input tokenization - I used `nltk.tokenize.casual.casual_tokenize`
* creation of TF-IDF vectorizer (`sklearn.feature_extraction.text.TfidfVectorizer`) with vocabulary that consists of tokenized user input, filtered by stop words
* building TF-IDF documents for corpus through vectorizer. Page titles are used as `pandas` indexes
* and finally... searching. Search is based on Principal Component Analysis (`PCA`)

### 2.7 How does this search work

First, `sklearn.decomposition.PCA` instance is created, and fitted:

```python
pca = PCA(n_components=1)
pca.fit(tfidf_docs.values)
```

We are only interested in one component - our search query - so `n_components` can stay as 1.

Next, PCA topic vectors are created through transformation:
```python
pca_topic_vectors = pca.transform(tfidf_docs.values)
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=[Q], index=index)
```

There is only one column - our search query. And indexes (rows) are corresponding to Wikipedia pages that we loaded.

Lastly, we calculate TF-IDF and PCA for our search query:
```python
tfidf_q = pd.DataFrame(tfidf_vectorizer.transform(raw_documents=[Q]).toarray())
pca_q = pca.transform(tfidf_q.values)
```

### 2.8 Finding the right answer

We now have all that we need to determine which page is closest.

```python
for page_title, row in pca_topic_vectors.iterrows():
    distance = abs(row[Q] - pca_q[0][0])
    if distance < min_distance:
        min_distance = distance
        min_topic = page_title
```

`min_topic` is the page title that is closest to the probable search result we are looking for.

### 2.9 Displaying the result

```python
found_page_title = min_topic
print('Found page: ', found_page_title)
print('Page content: ', pages.get(found_page_title)['text'])
```

### 2.10 Lookup preview

```
> bush cat
tfidf_docs = 
                              bush       cat
Creme Puff (cat)         0.000000  1.000000
Tuxedo Stan              0.000000  1.000000
Think Think and Ah Tsai  0.000000  1.000000
Catmando                 0.000000  1.000000
Freya (cat)              0.000000  1.000000
Gladstone (cat)          0.000000  1.000000
India (cat)              0.988341  0.152254
Hank the Cat             0.000000  1.000000
Nyan Cat                 0.000000  1.000000
Grumpy Cat               0.000000  1.000000

pca_topic_vectors = 
                          bush cat
Creme Puff (cat)        -0.130211
Tuxedo Stan             -0.130211
Think Think and Ah Tsai -0.130211
Catmando                -0.130211
Freya (cat)             -0.130211
Gladstone (cat)         -0.130211
India (cat)              1.171899
Hank the Cat            -0.130211
Nyan Cat                -0.130211
Grumpy Cat              -0.130211

pca_q =  [[1.00700294]] 

topic = "Creme Puff (cat)",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Tuxedo Stan",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Think Think and Ah Tsai",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Catmando",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Freya (cat)",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Gladstone (cat)",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "India (cat)",  avg = 1.1718994850439566,  pca_q = 1.0070029372017024,  distance = 0.16489654784225416, 
topic = "Hank the Cat",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Nyan Cat",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 
topic = "Grumpy Cat",  avg = -0.13021105389377297,  pca_q = 1.0070029372017024,  distance = 1.1372139910954755, 

min_distance =  0.16489654784225416
min_topic =  India (cat)

=============================
Found page:  India (cat)
Page content:  India "Willie" Bush (July 13, 1990 – January 4, 2009) 
was a black Shorthair cat owned by former U.S. President George W. Bush and First Lady Laura Bush...
```

## Summary

Implementation of semantic search through TF-IDF and PCA is satisfying, but unfortunately, is not scalable. Each search requires TF-IDF, and PCA vectors to be recreated which means that either:
(1) corpus is reloaded on each iteration 
(2) corpus is kept in memory all the time

This is all good when dealing with smaller datasets (tens, hundreds, thousands, tens of thousands, or hundreds of thousands) but becomes serve weakness when dealing with millions or billions of records.

Creators of search engines like Google know that and thus are using solutions like [inverted database indexes](https://en.wikipedia.org/wiki/Search_engine_indexing#Inverted_indices) to avoid this issue.

## Bibliography & Sources

* [[Misc] Wikitext documentation](https://www.mediawiki.org/wiki/Wikitext)
* [[GitHub] wikiextractor](https://github.com/attardi/wikiextractor)
* [[Python 3] glob module documentation](https://docs.python.org/3.7/library/glob.html#glob.glob)
* Hobson Lane, Cole Howard, Hannes Hapke - Natural Language Processing in Action
