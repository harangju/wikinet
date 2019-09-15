"""
Module: wiki

contains classes
- Dump
- Corpus
- Net
- Crawler
"""

import sys
import os
import bz2
import re
import math
import xml.etree.ElementTree as ET
import mwparserfromhell as mph
import networkx as nx
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import dionysus as d
import pandas as pd
import numpy as np

class Dump:
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
    years: list of int
        Years in the History section of a wikipedia page
        BC denoted as negative values
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
    def __init__(self, dump):
        self.dump = dump
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
    
    def __getitem__(self, index):
        doc = self.dump.load_page(self.names[index])
        return simple_preprocess(doc.strip_code())

class Net:
    """ Net is a wrapper for networkx.DiGraph.
    Uses dionysus for persistence homology.
    
    Attributes
    ----------
    name: string
        Name of network
    graph: networkx.DiGraph
        node name is name of wikipedia page
        'Year' attribute indicates year
    numbered: networkx.DiGraph (lazy)
        node name is an index (see nodes)
        'Year' is an index (see years)
    nodes: list (lazy)
        List of node names,
        indexed by node in numbered
    years: list (lazy)
        List of years,
        indexed by 'Year' attribute in numbered
    nodes_for_year: dict (lazy)
        Dictionary of {int year: [int node_index]}
        (see nodes)
    cliques: list of lists (lazy)
    filtration: dionysus.filtration (lazy)
    persistence: dionysus.reduced_matrix (lazy)
    barcodes: pandas.DataFrame (lazy)
    
    Methods
    -------
    build_graph(path)
    load_graph(path)
    save_graph(path)
    
    Static methods
    --------------        
    fill_empty_nodes()
    bft()
    filter()
    """
    def __init__(self, name='', graph=None, numbered=None,
                 nodes=[], years=[], nodes_for_year={},
                 cliques=[], filtration=None,
                 persistence=None, barcodes=None):
        self.name = name
        self.graph = graph
        self._numbered = numbered
        self._nodes = nodes
        self._years = years
        self._nodes_for_year = nodes_for_year
        self._cliques = cliques
        self._filtration = filtration
        self._persistence = persistence
        self._barcodes = barcodes
    
    def build_graph(self, dump, nodes=None, depth_goal=1, filter_top=True,
                    remove_isolates=True, add_years=True, fill_empty_years=True):
        """ Builds self.graph (networkx.Graph) from nodes
        Parameters
        ----------
        dump: wiki.Dump
        nodes: list of **strings**
        depth_goal: int
        filter_top: bool
        add_years: bool
        fill_empty_years: bool
        """
        self.graph = nx.DiGraph()
        print('wiki.Net: traversing Wikipedia...')
        Net.bft(self.graph, dump, nodes, depth_goal=depth_goal, 
                nodes=nodes, filter_top=filter_top)
        if remove_isolates:
            print('wiki.Net: removing isolates...')
            self.graph.remove_nodes_from(nx.isolates(self.graph))
        if add_years:
            print('wiki.Net: adding years...')
            for node in self.graph.nodes:
                dump.load_page(node)
                self.graph.nodes[node]['year'] = dump.years[0] if len(dump.years)>0 else []
        if fill_empty_years:
            print('wiki.Net: filling empty years...')
            nodes_filled = True
            while nodes_filled:
                nodes_filled = Net.fill_empty_nodes(self.graph, full_parents=True)
            nodes_filled = True
            while nodes_filled:
                nodes_filled = Net.fill_empty_nodes(self.graph, full_parents=False)
            for node in self.graph.nodes:
                if not self.graph.nodes[node]['year']:
                    self.graph.nodes[node]['year'] = 2020#math.inf
    
    def get_numbered(self):
        if self._numbered:
            return self._numbered
        else:
            self._numbered = nx.DiGraph()
            for node in self.graph.nodes:
                self._numbered.add_node(self.nodes.index(node),
                                        year = self.years.index(self.graph.nodes[node]['year']))
                self._numbered.add_edges_from([(self.nodes.index(node), self.nodes.index(succ))
                                               for succ in self.graph.successors(node)])
            return self._numbered
    numbered = property(get_numbered)
    
    def get_nodes(self):
        if self._nodes:
            return self._nodes
        else:
            self._nodes = [n for n in self.graph.nodes()]
            return self._nodes
    nodes = property(get_nodes)
    
    def get_years(self):
        if self._years:
            return self._years
        else:
            self._years = [self.graph.nodes[n]['year']
                           for n in self.graph.nodes()]
            self._years = sorted(list(set(self._years)))
            return self._years
    years = property(get_years)
    
    def get_nodes_for_year(self):
        if self._nodes_for_year:
            return self._nodes_for_year
        else:
            self._nodes_for_year = {year: [self.nodes.index(n)
                                           for n in self.nodes
                                           if self.graph.nodes[n]['year']==year]
                                    for year in self.years}
            return self._nodes_for_year
    nodes_for_year = property(get_nodes_for_year)
    
    def get_cliques(self):
        if self._cliques:
            return self._cliques
        else:
            self._cliques = list(nx.algorithms.clique.\
                                 enumerate_all_cliques(nx.Graph(self.numbered)))
            return self._cliques
    cliques = property(get_cliques)
    
    def get_filtration(self):
        if self._filtration != None:
            return self._filtration
        else:
            self._filtration = d.Filtration()
            nodes_so_far = []
            for year in self.years:
                nodes_now = self.nodes_for_year[year]
                nodes_so_far.extend(nodes_now)
                for clique in self.cliques:
                    if all([n in nodes_so_far for n in clique]):
                        self._filtration.append(d.Simplex(clique, year))
            self._filtration.sort()
            return self._filtration
    filtration = property(get_filtration)
    
    def get_persistence(self):
        if self._persistence:
            return self._persistence
        else:
            self._persistence = d.homology_persistence(self.filtration)
            return self._persistence
    persistence = property(get_persistence)
    
    def get_barcodes(self):
        if isinstance(self._barcodes, pd.DataFrame)\
            and len(self._barcodes.index) != 0:
            return self._barcodes
        else:
            self._barcodes = Net.compute_barcodes(self.filtration,
                                                  self.persistence,
                                                  self.graph, self.nodes)
            return self._barcodes
    barcodes = property(get_barcodes)
    
    def load_graph(self, path):
        self.graph = nx.read_gexf(path)
    
    def save_graph(self, path):
        nx.write_gexf(self.graph, path)
    
    @staticmethod
    def fill_empty_nodes(graph, full_parents=True):
        """
        Parameters
        ----------
        graph: networkx.DiGraph
        full_parents: bool
            whether to fill empty nodes that have
            all parents with non-empty 'year'
        Returns
        -------
        bool
            whether at least 1 empty node was filled
        """
        empty_nodes = [n for n in graph.nodes if not graph.nodes[n]['year']]
        for node in empty_nodes:
            years = [graph.nodes[p]['year'] for p in graph.predecessors(node)]
            if not years:
                continue
            if full_parents:
                if [] not in years:
                    graph.nodes[node]['year'] = max(years)
                    return True
            else:
                years_filtered = [y for y in years if y]
                if years_filtered:
                    graph.nodes[node]['year'] = max(years_filtered)
                    return True
        return False
    
    @staticmethod
    def bft(graph, dump, queue, depth_goal=1, nodes=None, filter_top=True):
        """ breadth-first traversal
        Parameters
        ----------
        graph: networkx.DiGraph
        dump: wiki.Dump
        queue: list of **strings**
            names of Wikipedia pages
        depth_goal: int
        nodes: list of **strings**
            names of Wikipedia pages
        filter_top: bool
        """
        queue = queue.copy()
        page_noload = []
        depth = 0
        depth_num_items = len(queue)
        depth_inc_pending = False
        print('wiki.Net: depth = ' + str(depth))
        while queue:
            name = queue.pop(0)
            sys.stdout.write("\rwiki.Net: len(queue) = " + str(len(queue)))
            sys.stdout.flush()
            depth_num_items -= 1
            if depth_num_items == 0:
                depth += 1
                print('\nwiki.Net: depth = ' + str(depth))
                depth_inc_pending = True
            page = dump.load_page(name, filter_top=filter_top)
            if not page:
                page_noload.append(name)
                continue
            links = [l for l in dump.article_links
                     if Net.filter(name, l, graph, nodes)]
            for link in links:
                graph.add_edge(link, name, weight=1)
                if link not in queue:
                    queue.append(link)
            if depth_inc_pending:
                depth_num_items = len(queue)
                depth_inc_pending = False
            if depth == depth_goal:
                print('')
                break
        return page_noload
    
    @staticmethod
    def filter(page, link, graph, nodes=None):
        if nodes and link not in nodes:
            return False
        if (page, link) in graph.edges:
            return False
        return True
    
    @staticmethod
    def compute_barcodes(f, m, graph, names):
        """ Uses dionysus filtration & persistence
        (in reduced matrix form) to compute barcodes
        Parameters
        ----------
        f: dionysus.Filtration
        m: dionysus.ReducedMatrix
            (see homology_persistence)
        names: list of strings
            names of node indices
        Returns
        -------
        barcodes: pandas.DataFrame
        """
        print('wiki.Net: computing barcodes... (skip negatives)')
        barcodes = []
        for i in range(len(m)):
            if m.pair(i) < i: continue
            sys.stdout.write("\rwiki.Net: barcode {}/{}".\
                             format(i+1,len(m)))
            sys.stdout.flush()
            dim = f[i].dimension()
            birth_year = int(f[i].data)
            birth_simplex = [names[s] for s in f[i]]
            birth_nodes = [n for n in birth_simplex
                           if graph.nodes[n]['year']==birth_year]
            if m.pair(i) != m.unpaired:
                death_year = int(f[m.pair(i)].data)
                death_simplex = [names[s] for s in f[m.pair(i)]]
                death_nodes = [n for n in death_simplex
                               if graph.nodes[n]['year']==death_year]
            else:
                death_year = np.inf
                death_simplex = []
                death_nodes = []
            barcodes.append([dim, birth_year, death_year,
                             birth_simplex, death_simplex,
                             birth_nodes, death_nodes])
        print('')
        barcodes.sort(key=lambda x: x[0])
        bar_data = pd.DataFrame(data=barcodes,
                                columns=['dim', 'birth', 'death',
                                         'birth simplex', 'death simplex',
                                         'birth nodes', 'death nodes'])
        return bar_data