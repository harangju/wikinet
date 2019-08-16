from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import sys
from .dump import WikiDump

class WikiCorpus:
    def __init__(self, path_xml, path_index):
        self.dump = WikiDump(path_xml, path_index)
        self.names = list(self.dump.idx.keys())
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < len(self.names):
            sys.stdout.write("\rCorpus index: " + str(self.i) + 
                             '/' + str(len(self.names)))
            sys.stdout.flush()
            doc = TaggedDocument(self.doc_at(self.i), [self.i])
            self.i += 1
            return doc
        else:
            raise StopIteration
    
    def doc_at(self, index):
        doc = self.dump.load_page(self.names[index])
        return simple_preprocess(doc.strip_code())