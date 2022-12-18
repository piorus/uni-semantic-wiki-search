# Project Report

## Title

Semantic Wikipedia Search written in Python

## Authors

Piotr Rusin

## Abstract

## 1. Introduction

### 1.1 Problem to solve

Construct a semantic search engine with a database based on Wikipedia (selected topic). Please use different methods to construct word vectors. Use Python libraries.

### 1.2 Selection of topic

As a topic I selected cats because Wikipedia stores information about a wide range of species, and related pages are frequently updated by cat lovers. 

### 1.3 Wikipedia page export

Wikipedia allows anyone to download its database, and - as it turned out - this download can also be constrained to chosen pages.

To make the whole process less of a burden, Wikipedia founders created a tool called [Export pages](https://en.wikipedia.org/wiki/Special%3aExport) 
which allows you to select a category you wish to export, and then it automatically pulls a current list of pages that are within this category. 

I used it to create an aggregated list of pages to export. This list is based on the following categories:

```
Category:Lists_of_cats
Category:Individual_cats
Category:Cat_folklore
Category:Cat_monuments
Category:Fictional_cats
```

A complete list of pages that this tool listed for export is located  [here](report/exported-wiki-pages.md). 

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

The unprocessed XML file that was exported as a result is located within this repository and is located [here](Wikipedia.xml).

## 2. Development

### 2.1 Parsing raw XML into JSON

I used a [wikiextractor](https://github.com/attardi/wikiextractor) Python library to convert pages from XML with content as [wikitext](https://www.mediawiki.org/wiki/Wikitext) into JSON with content as plaintext.

First, I installed it through pipenv:
```
pipenv install wikiextractor
```

Then, I converted pages to JSON by executing:
```
pipenv run wikiextractor Wikipedia.xml --json
```

As a result, a new `text` directory was created within the project root with two files `wiki_00`, and `wiki_01`:

![wikiextractor1.png](report%2Fwikiextractor1.png)

with the following content:

![wikiextractor2.png](report%2Fwikiextractor2.png)

### 2.2 Parsing wikiextractor output

wikiextractor saved the pages JSON in two separate files (useful for bigger exports). It saved us a lot of time that would otherwise had to be spent on loading & parsing XML, converting wikitext into plaintext, and exporting it into JSON.

This output still needs to be loaded in Python, and JSON string needs to be parsed into Python dictionary.

So I decided to create a generator that recursively iterates over wikiextractor output files, parse each line as JSON, and yields it as a result:
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

I will use this generator to prepare a corpus for the semantic search.

### 2.3 Building a corpus

Just a simple list comprehension that extract page content and append it into the corpus list:
```python
corpus = [page['text'].replace('\n','') for page in wiki_pages_generator()]
```

We also want to ensure that there are no duplicates:

```python
corpus = list(set(corpus))
```

### 2.4 Calculating TF-IDF vector

I used `TfidfVectorizer` from `scikit-learn` module for TF-IDF calculation with a stopwords, and tokenizer from `nltk`:

```python
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')

tfidf_vectorizer = TfidfVectorizer(
    tokenizer=nltk.tokenize.casual.casual_tokenize,
    stop_words=stop_words
)
tfidf_docs = pd.DataFrame(tfidf_vectorizer.fit_transform(corpus).toarray())
id_words = [(i, w) for (w, i) in tfidf_vectorizer.vocabulary_.items()]
tfidf_docs.columns = list(zip(*sorted(id_words)))[1]
```

And ended up with `262 rows x 22724 columns` matrix.

## Summary

## Bibliography & Sources

* [[Misc] Wikitext documentation](https://www.mediawiki.org/wiki/Wikitext)
* [[GitHub] wikiextractor](https://github.com/attardi/wikiextractor)
* [[Python 3] glob module documentation](https://docs.python.org/3.7/library/glob.html#glob.glob)
* [[YouTube] How to Build a Semantic Search System - Trey Grainger, Lucidworks](https://www.youtube.com/watch?v=4fMZnunTRF8)

The report should contain all relevant information describing the project. 
Please write it as if you were trying to explain to your colleagues the problem you solved. 
Please divide it into at least three sections (there may be more depending on your needs): 
0. Title, authors, abstract 
1. Introduction – a section describing the problem you were solving
2. Development - a section describing your solution. Here you can discuss the most important parts of the program or describe the data analysis.
3. Summary - in this section drawing conclusions related to the project: analysis of the results, information on whether the project was successful, if not, why?
4. Bibliography - items you used in the project and preparation of the report. Each item in the bibliography should be quoted in the text. 

Please also remember that the use of fragments of texts / code without giving the source is plagiarism and it is a crime: 
https://pl.wikipedia.org/wiki/Plagiat%%% 
Projects will be placed on a public website, so if you do not want your names to be there, please anonymise the names, e.g. initials, before sending.
