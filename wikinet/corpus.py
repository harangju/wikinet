import gensim.utils as gu
import gensim.models as gm

from .dump import Dump

__all__ = ['Corpus']

class Corpus:
    """``Corpus`` is an ``iterable`` & an ``iterator``
    that uses ``Dump`` to iterate through articles.

    dump: wiki.Dump
    output: string
        'doc' for array of documents
        'tag' for TaggedDocument(doc, [self.i])
        'bow' for bag of words [(int, int)]
    dct: gensim.corpus.Dictionary
        used to create BoW representation
    """
    def __init__(self, dump, output='doc', dct=None, load_index=True):
        self.dump = dump
        if load_index:
            self.names = list(self.dump.idx.keys())
        self.output = output
        self.dct = dct

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.names):
            sys.stdout.write(
                "\rCorpus index: " + str(self.i+1) +
                '/' + str(len(self.names))
            )
            sys.stdout.flush()
            if self.output == 'doc':
                doc = self[self.i]
            elif self.output == 'tag':
                doc = gm.doc2vec.TaggedDocument(self[self.i], [self.i])
            elif self.output == 'bow':
                doc = self.dct.doc2bow(self[self.i])
            self.i += 1
            return doc
        else:
            raise StopIteration

    def __getitem__(self, index):
        doc = self.dump.load_page(self.names[index])
        return gu.simple_preprocess(doc.strip_code())
