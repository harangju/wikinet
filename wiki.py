import sys
import os
import bz2
import re
import xml.etree.ElementTree as ET
import mwparserfromhell as mph
import networkx as nx
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

class Dump():
    def __init__(self, path_xml, path_idx):
        self._idx = {}
        self._article_links = []
        self._links = []
        self._page = None
        self.path_xml = path_xml
        self.path_idx = path_idx
        self.offset_max = 0
        self.cache = (0, None) # offset, cache
        
    def get_idx(self):
        if self._idx:
            return self._idx
        else:
            print('Dump: Loading index...')
            with bz2.BZ2File(self.path_idx, 'rb') as file:
                lines = [line for line in file]
            block_end = os.path.getsize(self.path_xml)
            offset_prev = block_end
            for line in reversed(lines):
                offset, pid, name = line.strip().split(b':', 2)
                offset, pid, name = (int(offset), int(pid), name.decode('utf8'))
                block_end = offset_prev if offset < offset_prev else block_end
                self._idx[name] = (offset, pid, block_end-offset)
                offset_prev = offset
            self.offset_max = max([x[0] for x in self._idx.values()])
            print('Dump: Loaded.')
            return self._idx
    idx = property(get_idx)
    
    def get_links(self):
        if self._links:
            return self._links
        elif self.page:
            self._links = [x.title for x in self.page.filter_wikilinks()]
            return self._links
    links = property(get_links)
    
    def get_article_links(self):
        if self._article_links:
            return self._article_links
        elif self.page:
            self._article_links = [x.title for x in self.page.filter_wikilinks()]
            return self._article_links
    article_links = property(get_article_links)
    
    def get_page(self):
        return self._page
    
    def set_page(self, page):
        self._page = page
        self._links = []
    page = property(get_page, set_page)
    
    def load_page(self, page_name, filter_top=False):
        if page_name not in self.idx.keys():
            self.page = None
            return
        offset, pid, block_size = self.idx[page_name]
        if offset == self.cache[0]:
            root = self.cache[1]
        else:
            xml = Dump.fetch_block(self.path_xml, offset, block_size)
            xml = b'<mediawiki>' + xml + b'</mediawiki>'*(offset != self.offset_max)
            root = ET.fromstring(xml)
            self.cache = (offset, root)
        text = Dump.search_id(root, pid)
        text = Dump.filter_top_section(text) if filter_top else text
        self.page = mph.parse(text)
        return self.page
    
    @staticmethod
    def fetch_block(path, offset, block_size):
        with open(path, 'rb') as file:
            file.seek(offset)
            return bz2.decompress(file.read(block_size))
    
    @staticmethod
    def search_id(root, pid):
        for page in root.iter('page'):
            if pid == int(page.find('id').text):
                return page.find('revision').find('text').text
    
    @staticmethod
    def filter_top_section(text):
        head = re.search(r'==.*?==', text)
        idx = head.span(0)[0] if head else len(text)
        return text[:idx] #(text[:idx], text[idx:])

class Corpus:
    def __init__(self, path_xml, path_index):
        self.dump = Dump(path_xml, path_index)
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
    
class Crawler():
    @staticmethod
    def bfs(graph, dump, queue, depth_goal=1, nodes=None):
        # all elements in queue & nodes should be of type string
        depth = 0
        depth_num_items = len(queue)
        depth_inc_pending = False
        print('Depth: ' + str(depth))
        while queue:
            name = queue.pop(0)
            depth_num_items -= 1
            if depth_num_items == 0:
                depth += 1
                print('Depth: ' + str(depth))
                depth_inc_pending = True
            page = dump.load_page(name, filter_top=True)
            if not page:
                continue
            links = [Crawler.parse(l) for l in dump.links]
            links = [l for l in links\
                     if Crawler.filter(name, l, graph, nodes)]
            for link in links:
                graph.add_edge(link, name, weight=1)
                queue.append(link)
            if depth_inc_pending:
                depth_num_items = len(queue)
                depth_inc_pending = False
            if depth == depth_goal:
                break
    
    @staticmethod
    def filter(page, link, graph, nodes=None):
        if nodes and link not in nodes:
            return False
        if (page, link) in graph.edges:
            return False
        return True
    
    @staticmethod
    def parse(link):
         return str(link).split('#')[0].capitalize()
    
    @staticmethod
    def filter(page, link, graph, nodes=None):
        if nodes and link not in nodes:
            return False
        if (page, link) in graph.edges:
            return False
        return True
    
    @staticmethod
    def parse(link):
         return str(link).split('#')[0].capitalize()