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
    """Dump loads and parses dumps from wikipedia.
    
    Attributes
    ----------
    idx: dictionary
        Loaded index file as {'page_name': (byte offset, page id, block size)}
        Cached. Lazily loaded (when needed).
    links: list of strings
        All links.
    article_links: list of strings
        Article links (not files, categories, etc.)
    page: mwparserfromhell.wikicode
        Current loaded wiki page
    path_xml: string
        Path to the zipped XML dump file.
    path_idx: string
        Path to the zipped index file.
    offset_max: int
        Maximum offset. Set as the size of the zipped dump.
    cache: xml.etree.ElementTree.Node
        Cache of the XML tree in current block
    
    Methods
    -------
    load_page(page_name)
        Loads page with page_name
    
    Static methods
    --------------
    fetch_block(path, offset, block_size) -> string
        Fetches block of bytes at offset in the zipped dump at path.
        Returns uncompressed text.
    search_id(root, pid) -> string
        Returns the text of the page with id pid
    filter_top_section(text) -> string
        Returns the top section of text,
        where the first header has the form '==Heading=='
    get_history(page) -> string
        Returns the text of the history section.
        Returns None if not found.
    filter_years(text) -> list of integers
        Filters the years from text.
    """
    
    def __init__(self, path_xml, path_idx):
        self._idx = {}
        self._links = []
        self._article_links = []
        self._years = []
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
            self._links = [str(x.title).split('#')[0].capitalize()
                           for x in self.page.filter_wikilinks()]
            return self._links
        else:
            return self._links
    links = property(get_links)
    
    def get_article_links(self):
        if self._article_links:
            return self._article_links
        elif self.links:
            self._article_links = [x for x in self.links if ':' not in x]
            return self._article_links
        else:
            return self._article_links
    article_links = property(get_article_links)
    
    def get_years(self):
        if self._years:
            return self._years
        elif self.page:
            history = Dump.get_history(self.page)
            self._years = Dump.filter_years(history) if history else []
            return self._years
        else:
            return self._years
    years = property(get_years)
    
    def get_page(self):
        return self._page
    
    def set_page(self, page):
        self._page = page
        self._links = []
        self._article_links = []
        self._years = []
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
        self.page = mph.parse(text, skip_style_tags = True)
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
    
    @staticmethod
    def get_history(page):
        headings = page.filter_headings()
        idx = [i for i, head in enumerate(headings) 
                       if 'History' in head]
        if not idx:
            return
        sections = page.get_sections(include_headings=True)
        history = str(sections[idx[0]+1].strip_code())
        return history
    
    @staticmethod
    def filter_years(text):
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        prepositions = ['about', 'around', 'after', 'at', 'as',
                        'approximately', 'before', 'between', 'by',
                        'during', 'from', 'in', 'near', 'past',
                        'since', 'until', 'within'] # removed: on
        conjugations = ['and']
        articles = ['the']
        times = ['early', 'mid', 'late']
        patterns = months + prepositions + conjugations + articles + times
        re_string = r'\b(' + '|'.join(patterns) + r')\b(\s|-)\b([0-9]{3,4})s?\b\s?(BCE|BC)?'
        years = [int(match.group(3)) * (-2*bool(match.group(4))+1)
                for match in re.finditer(re_string, text, re.IGNORECASE)]
        re_string = r'([0-9]{1,2})th century\s?(BCE|BC)?'
        centuries = [(int(match.group(1)) * 100 - 100) * (-2*bool(match.group(2))+1)
                     for match in re.finditer(re_string, text, re.IGNORECASE)]
        return sorted(years + centuries)

class Corpus:
    """Corpus is an iterable & an iterator
    that uses Dump to iterate through articles.
    """
    def __init__(self, path_xml, path_index):
        # init with dump, not paths
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
        queue = queue.copy()
        page_noload = []
        depth = 0
        depth_num_items = len(queue)
        depth_inc_pending = False
        print('Depth: ' + str(depth))
        while queue:
            name = queue.pop(0)
            sys.stdout.write("\rCrawler: len(queue) = " + str(len(queue)))
            sys.stdout.flush()
            depth_num_items -= 1
            if depth_num_items == 0:
                depth += 1
                print('\nDepth: ' + str(depth))
                depth_inc_pending = True
            page = dump.load_page(name, filter_top=True)
            if not page:
                page_noload.append(name)
                continue
            links = [l for l in dump.article_links
                     if Crawler.filter(name, l, graph, nodes)]
            for link in links:
                graph.add_edge(link, name, weight=1)
                if link not in queue:
                    queue.append(link)
            if depth_inc_pending:
                depth_num_items = len(queue)
                depth_inc_pending = False
            if depth == depth_goal:
                break
        return page_noload
    
    @staticmethod
    def filter(page, link, graph, nodes=None):
        if nodes and link not in nodes:
            return False
        if (page, link) in graph.edges:
            return False
        return True