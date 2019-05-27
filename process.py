import numpy as np
from sklearn import svm

from gensim.corpora import Dictionary
from gensim.utils import tokenize
from gensim.summarization.textcleaner import get_sentences
from gensim.models import TfidfModel


def main():
    #-------------------------- Text processing
    with open('text.txt', 'r') as f:
        raw = f.read()

    print('\n----------------------\n\tRaw text\n----------------------')
    print(raw)


    sentences = list(get_sentences(raw))
    tokens = list([list(tokenize(s)) for s in sentences])
    print('\n----------------------\n\tDictionary\n----------------------')
    d = Dictionary(tokens)
    print(d)

    t = tokens[2]

    print('\n----------------------\n\t"Bag of Words"\n----------------------')
    print('{0} -> {1}'.format(t, d.doc2bow(t)))

    corpus = list([d.doc2bow(t) for t in tokens])
    tfidf = TfidfModel(corpus)

    print('\n----------------------\n\tTFIDF\n----------------------')
    print('{0} -> {1}'.format(t, list(tfidf[corpus[2]])))

    results = [tfidf[c] for c in corpus]

    #------------------------------ Training
    # Normally, we would take this `y` data from csv file, where we have raw text and label together.
    # Cheating a bit, so we label anything with `Duke` inside as a label `1`
    y = [1 if 'Duke' in t else 0 for t in tokens]
    x = np.zeros((len(results), len(d)))

    for i, r in enumerate(results):
        for j, v in r:
            x[(i, j)] = v


    model = svm.LinearSVC()
    model.fit(x, y)


    #------------------------------ Testing

    print('\n----------------------\n\tPredicting labels\n----------------------')


    s1 = 'Our Duke Vytautas'
    s2 = 'Our King Vytautas'
    s3 = 'You know nothing, John Snow'
    s4 = 'Duke Nukem 3D'

    def prepare(s):
        return tfidf[d.doc2bow(tokenize(s))]

    results = [prepare(s) for s in [s1, s2, s3, s4]]

    x = np.zeros((len(results), len(d)))

    for i, r in enumerate(results):
        for j, v in r:
            x[(i, j)] = v

    for o, r in zip([s1, s2, s3, s4], model.predict(x)):
        print('{0}: {1}'.format(o, r))


if __name__ == '__main__':
    main()
