import os
import re
import sys
import bz2
import bct
import math
import random
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import dionysus as d
import networkx as nx
import cpnet as cp
import gensim.utils as gu
import gensim.models as gm
import gensim.matutils as gmat
import gensim.parsing.preprocessing as gpp
import mwparserfromhell as mph
import xml.etree.ElementTree as ET
import sklearn.metrics.pairwise as smp

class Dump:
    """``Dump`` loads and parses dumps from wikipedia from
    ``path_xml`` with index ``path_idx``.
    
    Attributes
    ----------
    idx: dictionary
        ``{'page_name': (byte offset, page id, block size)}``
        Cached. Lazy.
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
    """
    MAX_YEAR = 2020
    
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
        
    @property
    def idx(self):
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
    
    @property
    def links(self):
        if self._links:
            return self._links
        elif self.page:
            self._links = [str(x.title) for x in self.page.filter_wikilinks()]
            self._links = [link.split('#')[0] for link in self._links]
            self._links = [link.split(' ') for link in self._links]
            self._links = [[words[0].capitalize()] + words[1:] for words in self._links]
            self._links = [' '.join(words) for words in self._links]
            return self._links
        else:
            return self._links
    
    @property
    def article_links(self):
        if self._article_links:
            return self._article_links
        elif self.links:
            self._article_links = [x for x in self.links if ':' not in x]
            return self._article_links
        else:
            return self._article_links
    
    @property
    def years(self):
        if self._years:
            return self._years
        elif self.page:
            history = Dump.get_history(self.page)
            top = self.page.get_sections()[0].strip_code()
            self._years = Dump.filter_years(top + history)
            return self._years
        else:
            return self._years
    
    @property
    def page(self):
        return self._page
    
    @page.setter
    def page(self, page):
        self._page = page
        self._links = []
        self._article_links = []
        self._years = []
    
    def load_page(self, page_name, filter_top=False):
        """Loads & returs page (``mwparserfromhell.wikicode``)
        named ``page_name`` from dump file. Returns only the
        top section if ``filter_top``.
        """
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
        if self.page and 'REDIRECT' in self.page.strip_code():
            redirect = self.page.filter_wikilinks()[0].title
            return self.load_page(str(redirect))
        else:
            return self.page
    
    @staticmethod
    def fetch_block(path, offset, block_size):
        """ Fetches block of ``block_size`` (``int``) bytes
        at ``offset`` (``int``) in the zipped dump at 
        ``path`` (``string``) and returns the uncompressed
        text (``string``).
        """
        with open(path, 'rb') as file:
            file.seek(offset)
            return bz2.decompress(file.read(block_size))
    
    @staticmethod
    def search_id(root, pid):
        """Returns the text of the page with id ``pid``"""
        for page in root.iter('page'):
            if pid == int(page.find('id').text):
                return page.find('revision').find('text').text
    
    @staticmethod
    def filter_top_section(text):
        """Returns the top section of text,
        where the first header has the form ``==Heading==``
        """
        head = re.search(r'==.*?==', text)
        idx = head.span(0)[0] if head else len(text)
        return text[:idx] #(text[:idx], text[idx:])
    
    @staticmethod
    def get_history(page):
        """Returns the text of the history section.
        Returns ``""`` if not found.
        """
        headings = page.filter_headings()
        idx = [i for i, head in enumerate(headings) 
                       if 'History' in head or 'history' in head]
        if not idx:
            return ""
        sections = page.get_sections(include_headings=True)
        history = str(sections[idx[0]+1].strip_code())
        return history
    
    @staticmethod
    def filter_years(text):
        """Filters the years from text."""
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        prepositions = ['around', 'after', 'at', 'as',
                        'approximately', 'before', 'between', 'by',
                        'during', 'from', 'in', 'near', 'past',
                        'since', 'until', 'within'] # removed: about, on
        conjugations = ['and']
        articles = ['the']
        times = ['early', 'mid', 'late']
        patterns = months + prepositions + conjugations + articles + times
        re_string = r'\b(' + '|'.join(patterns) + r')\b(\s|-)\b([0-9]{3,4})s?\b(?i)(?!\sMYA)\s?(BCE|BC)?'
        years = [int(match.group(3)) * (-2*bool(match.group(4))+1)
                for match in re.finditer(re_string, text, re.IGNORECASE)]
        re_string = r'([0-9]{1,2})(st|nd|rd|th) century\s?(BCE|BC)?'
        centuries = [(int(match.group(1)) * 100 - 100) * (-2*bool(match.group(2))+1)
                     for match in re.finditer(re_string, text, re.IGNORECASE)]
        years += centuries
        years = [y for y in years if y<Dump.MAX_YEAR]
        return sorted(years + centuries)

class Corpus:
    """``Corpus`` is an ``iterable`` & an ``iterator``
    that uses ``Dump`` to iterate through articles.
    
    Parameters
    ----------
    dump: wiki.Dump
    output: string
        'doc' for array of documents
        'tag' for TaggedDocument(doc, [self.i])
        'bow' for bag of words [(int, int)]
    dct: gensim.corpus.Dictionary
        used to create BoW representation
    """
    def __init__(self, dump, output='doc', dct=None):
        self.dump = dump
        self.names = list(self.dump.idx.keys())
        self.output = output
        self.dct = dct
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < len(self.names):
            sys.stdout.write("\rCorpus index: " + str(self.i+1) + 
                             '/' + str(len(self.names)))
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
    
class GraphContainer():
    """
    
    Attributes
    ----------
    graph: networkx.DiGraph
        node name is name of wikipedia page
        ``year`` attribute indicates year
    numbered: networkx.DiGraph
        node name is an index (see nodes),
        ``year`` is an index (see years), lazy
    nodes: list
        List of node names,
        indexed by node in ``numbered``, lazy
    years: list
        List of years,
        indexed by ``year`` in ``numbered``, lazy
    nodes_for_year: dict
        ``{int year: [int node_index]}``, lazy
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._numbered = None
        self._nodes = []
        self._years = []
        self._nodes_for_year = {}
    
    @property
    def numbered(self):
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
    
    @property
    def nodes(self):
        if self._nodes:
            return self._nodes
        else:
            self._nodes = list(self.graph.nodes)
            return self._nodes
    
    @property
    def years(self):
        if self._years:
            return self._years
        else:
            self._years = [self.graph.nodes[n]['year']
                           for n in self.graph.nodes()]
            self._years = sorted(list(set(self._years)))
            return self._years
    
    @property
    def nodes_for_year(self):
        if self._nodes_for_year:
            return self._nodes_for_year
        else:
            self._nodes_for_year = {year: [list(self.graph.nodes).index(n)
                                           for n in self.graph.nodes
                                           if self.graph.nodes[n]['year']==year]
                                    for year in self.years}
            return self._nodes_for_year
    
class PersistentHomology(GraphContainer):
    """
    
    Attributes
    ----------
    cliques: list of lists
        lazy
    filtration: dionysus.filtration
        lazy
    persistence: dionysus.reduced_matrix
        lazy
    barcodes: pandas.DataFrame
        lazy
    """
    
    def __init__(self):
        GraphContainer.__init__(self)
        self._cliques = None
        self._filtration = None
        self._persistence = None
        self._barcodes = None
    
    @property
    def cliques(self):
        if self._cliques:
            return self._cliques
        else:
            self._cliques = list(nx.algorithms.clique.\
                                 enumerate_all_cliques(nx.Graph(self.numbered)))
            return self._cliques
    
    @property
    def filtration(self):
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
    
    @property
    def persistence(self):
        if self._persistence:
            return self._persistence
        else:
            self._persistence = d.homology_persistence(self.filtration)
            return self._persistence
    
    @property
    def barcodes(self):
        if isinstance(self._barcodes, pd.DataFrame)\
            and len(self._barcodes.index) != 0:
            return self._barcodes
        else:
            self._barcodes = PersistentHomology.compute_barcodes(
                self.filtration, self.persistence, self.graph, self.nodes)
            return self._barcodes
    
    @staticmethod
    def compute_barcodes(f, m, graph, names):
        """Uses dionysus filtration & persistence
        (in reduced matrix form) to compute barcodes.
        
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
        node_list = list(graph.nodes)
        barcodes = []
        for i, c in enumerate(m):
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
            pair = m.pair(i) if m.pair(i) != m.unpaired else np.inf
            chain = m[pair] if pair != np.inf else m[i]
            simp_comp = [f[entry.index] for entry in chain]
            nodes = [node_list[idx] for simplex in simp_comp for idx in simplex]
            barcodes.append([dim, birth_year, death_year, death_year-birth_year,
                             birth_simplex, death_simplex,
                             birth_nodes, death_nodes, list(set(nodes))])
        print('')
        barcodes.sort(key=lambda x: x[0])
        bar_data = pd.DataFrame(data=barcodes,
                                columns=['dim', 'birth', 'death', 'lifetime',
                                         'birth simplex', 'death simplex',
                                         'birth nodes', 'death nodes',
                                         'homology nodes'])
        return bar_data

class Net(PersistentHomology):
    """``Net`` is a wrapper for ``networkx.DiGraph``.
    Uses ``dionysus`` for persistence homology.
    
    Attributes
    ----------
    tfidf: scipy.sparse.csc.csc_matrix
        sparse column matrix of tfidfs,
        ordered by nodes, also stored in
        ```self.graph.graph['tfidf']```, lazy
    MAX_YEAR: int
        ``year = MAX_YEAR (2020)`` for nodes with parents 
        without years
    YEAR_FILLED_DELTA: int
        ``year = year of parents + YEAR_FILLED_DELTA (1)``
    """
    MAX_YEAR = 2020
    YEAR_FILLED_DELTA = 1
    
    def __init__(self, path_graph='', path_barcodes=''):
        PersistentHomology.__init__(self)
        self._tfidf = None
        if path_graph:
            self.load_graph(path_graph)
        if path_barcodes:
            self.load_barcodes(path_barcodes)
    
    @property
    def tfidf(self):
        if self._tfidf:
            return self._tfidf
        elif 'tfidf' in self.graph.graph.keys():
            self._tfidf = self.graph.graph['tfidf']
            return self._tfidf
    
    def build_graph(self, name='', dump=None, nodes=None, depth_goal=1,
                    filter_top=True, remove_isolates=True, add_years=True,
                    fill_empty_years=True, model=None, dct=None,
                    compute_core_periphery=True, compute_communities=True,
                    compute_community_cores=True):
        """ Builds ``self.graph`` (``networkx.Graph``) from nodes (``list``
        of ``string``). Set ``model`` (from ``gensim``) and ``dct``
        (``gensim.corpora.Dictionary``) for weighted edges.
        """
        self.graph = nx.DiGraph()
        self.graph.name = name
        if not dump:
            raise AttributeError('wiki.Net: Provide wiki.Dump object.')
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
            self.graph.graph['num_years'] = sum(
                [bool(y) for y in nx.get_node_attributes(self.graph, 'year').values()]
            )
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
                    self.graph.nodes[node]['year'] = Net.MAX_YEAR
        if model and dct:
            print('wiki.Net: calculating weights...')
            self.graph.graph['tfidf'] = Net.compute_tfidf(self.nodes, dump, model, dct)
            Net.set_weights(self.graph)
        if compute_core_periphery:
            print('wiki.Net: computing core-periphery...')
            Net.assign_core_periphery(self.graph)
        if compute_communities:
            print('wiki.Net: computing communities...')
            Net.assign_communities(self.graph)
        if compute_community_cores:
            print('wiki.Net: computing cores within communities...')
            Net.assign_cores_to_communities(self.graph)
    
    def load_graph(self, path):
        """Loads ``graph`` from ``path``.
        If ``filename.gexf`` then read as ``gexf``.
        Else, use ``pickle``."""
        if path.split('.')[-1]=='gexf':
            self.graph = nx.read_gexf(path)
        else:
            self.graph = nx.read_gpickle(path)
    
    def save_graph(self, path):
        """Saves ``graph`` at ``path``.
        If ``filename.gexf`` then save as ``gexf``.
        Else, use ``pickle``."""
        if path.split('.')[-1]=='gexf':
            nx.write_gexf(self.graph, path)
        else:
            nx.write_gpickle(self.graph, path)
    
    def load_barcodes(self, path):
        """Loads ``barcodes`` from ``pickle``."""
        self._barcodes = pickle.load(open(path, 'rb'))
    
    def save_barcodes(self, path):
        """Saves ``barcodes`` as ``pickle``."""
        pickle.dump(self.barcodes, open(path, 'wb'))
    
    def randomize(self, null_type,
                  compute_core_periphery=True, compute_communities=True,
                  compute_community_cores=True):
        """Returns a new ``wiki.Net`` with a randomized 
        copy of ``graph``. Set ``null_type`` as one of
        ``'year'``, ``'target'``.
        """
        network = Net()
        network.graph = self.graph.copy()
        if null_type == 'year':
            years = list(nx.get_node_attributes(network.graph, 'year').values())
            random.shuffle(years)
            for node in network.graph.nodes:
                network.graph.nodes[node]['year'] = years.pop()
        elif null_type == 'target':
            nodes = list(network.graph.nodes)
            for s, t in self.graph.edges:
                network.graph.remove_edge(s, t)
                nodes.remove(t)
                network.graph.add_edge(s, random.choice(nodes),
                                       weight=self.graph[s][t]['weight'])
                nodes.append(t)
        elif null_type == 'source':
            nodes = list(network.graph.nodes)
            for s, t in self.graph.edges:
                network.graph.remove_edge(s, t)
                nodes.remove(s)
                network.graph.add_edge(random.choice(nodes), t,
                                       weight=self.graph[s][t]['weight'])
                nodes.append(s)
        if compute_core_periphery:
            print('wiki.Net: computing core-periphery...')
            Net.assign_core_periphery(network.graph)
        if compute_communities:
            print('wiki.Net: computing communities...')
            Net.assign_communities(network.graph)
        if compute_community_cores:
            print('wiki.Net: computing cores within communities...')
            Net.assign_cores_to_communities(self.graph)
        return network
    
    @staticmethod
    def fill_empty_nodes(graph, full_parents=True):
        """ Fills nodes without ``year`` with the ``year`` of parents
        
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
                    graph.nodes[node]['year'] = max(years) \
                                                + Net.YEAR_FILLED_DELTA
                    return True
            else:
                years_filtered = [y for y in years if y]
                if years_filtered:
                    graph.nodes[node]['year'] = max(years_filtered) \
                                                + Net.YEAR_FILLED_DELTA
                    return True
        return False
    
    @staticmethod
    def bft(graph, dump, queue, depth_goal=1, nodes=None, filter_top=True):
        """Breadth-first traversal of hyperlink graph.
        
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
            if depth == depth_goal: break
            page = dump.load_page(name, filter_top=filter_top)
            if not page: continue
            links = [l for l in dump.article_links
                     if Net.filter(name, l, graph, nodes)]
            for link in links:
                graph.add_edge(link, name, weight=1)
                if link not in queue:
                    queue.append(link)
            if depth_inc_pending:
                depth_num_items = len(queue)
                depth_inc_pending = False
    
    @staticmethod
    def filter(page, link, graph, nodes=None):
        """Filter out links"""
        if nodes and link not in nodes:
            return False
        if (page, link) in graph.edges:
            return False
        return True
    
    @staticmethod
    def compute_tfidf(nodes, dump, model, dct):
        """Compute tf-idf of pages with titles in ``nodes``.
        
        Parameters
        ----------
        nodes: list of nodes
        dump: wiki.Dump
        model: gensim.modes.tfidfmodel.TfidfModel
        dct: gensim.corpora.Dictionary
        
        Returns
        -------
        vecs: scipy.sparse.csc.csc_matrix
        """
        pages = [dump.load_page(page) for page in nodes]
        bows = [model[dct.doc2bow(gu.simple_preprocess(page.strip_code()))]
                if page else []
                for page in pages]
        return gmat.corpus2csc(bows)
    
    @staticmethod
    def set_weights(graph):
        """Set the weights of graph (``networkx.DiGraph``) as
        the cosine similarity between ``graph.graph['tf-idf']]``
        vectors of nodes."""
        vecs = graph.graph['tfidf']
        for n1, n2 in graph.edges:
            v1 = vecs[:,list(graph.nodes).index(n1)].transpose()
            v2 = vecs[:,list(graph.nodes).index(n2)].transpose()
            graph[n1][n2]['weight'] = smp.cosine_similarity(X=v1, Y=v2)[0,0]
    
    @staticmethod
    def assign_core_periphery(graph):
        """ Compute core-periphery of ``graph`` (``nx.DiGraph``;
        converted to symmetric ``nx.Graph``).
        Assign ``core`` as ``1`` or ``0`` to each node.
        Assign ``coreness`` to ``graph``.
        See ``core_periphery_dir()`` in ``bctpy``.
        """
        # borgatti-everett
        be = bct.core_periphery_dir(nx.convert_matrix.to_numpy_array(graph))
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['core_be'] = be[0][i]
        graph.graph['coreness_be'] = be[1]
        # rombach
        rb = cp.Rombach()
        rb.detect(graph)
        if rb.get_coreness() != 0:
            for node, coreness in rb.get_coreness().items():
                graph.nodes[node]['core_rb'] = coreness
            graph.graph['coreness_rb'] = rb.score()[0]
        else:
            for node in graph.nodes:
                graph.nodes[node]['core_rb'] = 0
            graph.graph['coreness_rb'] = 0
    
    @staticmethod
    def assign_communities(graph):
        """ Compute modular communities of ``graph`` (``nx.DiGraph``).
        Assign community number ``community`` to each node.
        Assign ``modularity`` to ``graph``.
        See ``greedy_modularity_communities`` in ``networkx``.
        """
        communities = nx.algorithms.community\
                        .greedy_modularity_communities(nx.Graph(graph))
        for node in graph.nodes:
            graph.nodes[node]['community'] = [i for i,c in enumerate(communities)
                                              if node in c][0]
        graph.graph['modularity'] = nx.algorithms.community.quality\
                                      .modularity(nx.Graph(graph),
                                                  communities)
    
    @staticmethod
    def assign_cores_to_communities(graph):
        """"""
        num_comm = max([graph.nodes[n]['community'] for n in graph.nodes])
        community_coreness_be = {i: 0 for i in range(num_comm)}
        community_coreness_rb = {i: 0 for i in range(num_comm)}
        for i in range(num_comm+1):
            community = [n for n in graph.nodes if graph.nodes[n]['community']==i]
            subgraph = graph.subgraph(community).copy()
            matrix = nx.convert_matrix.to_numpy_array(subgraph)
            if (matrix.size>1) & (np.sum(matrix)>0):
                # borgatti-everett
                be = bct.core_periphery_dir(matrix)
                # rombach
                rb = cp.Rombach()
                rb.detect(subgraph)
                # assign
                community_coreness_be[i] = be[1]
                community_coreness_rb[i] = rb.score()[0]
                cp_rb = rb.get_coreness()
                for j, node in enumerate(subgraph.nodes):
                    graph.nodes[node]['community_core_be'] = be[0][j]
                    graph.nodes[node]['community_core_rb'] = cp_rb[node]
            else:
                community_coreness_be[i] = 0
                community_coreness_rb[i] = 0
                for j, node in enumerate(subgraph.nodes):
                    graph.nodes[node]['community_core_be'] = 1
                    graph.nodes[node]['community_core_rb'] = 1
        graph.graph['community_coreness_be'] = community_coreness_be
        graph.graph['community_coreness_rb'] = community_coreness_rb

class Model(PersistentHomology):
    """
    
    Attributes
    ----------
    graph: networkx.DiGraph
    graph_parent: networkx.DiGraph
    vectors: scipy.sparse.csc_matrix
    vectors_parent: scipy.sparse.csc_matrix
    seeds: {node string: [scipy.sparse.csc_matrix]}
    thresholds: {node string: [float]}
    year: int
    record: pandas.DataFrame
        record of evolution
    year_start: int
    n_seeds: int
        number of seeds per node
    point, insert, delete: tuple
        See ``mutate()``.
    rvs: lambda n->float
        random values for point mutations & insertions
    dct: gensim.corpora.dictionary
    create: lambda n-> float 
        thresholds of cosine similarity with parent
        for node creation
    crossover: float
        threshold of cosine similarity with parent
        for crossing over nodes
    """
    
    def __init__(self, graph_parent, vectors_parent, year_start, start_nodes,
                 n_seeds, dct, point, insert, delete, rvs,
                 create, crossover=None):
        """
        
        Parameters
        ----------
        start_nodes: lambda wiki.Model -> list(networkx.Nodes)
        """
        PersistentHomology.__init__(self)
        self.graph_parent = graph_parent
        self.vectors_parent = vectors_parent
        self.year_start = year_start
        self.year = year_start
        self.seeds = {}
        self.thresholds = {}
        self.record = pd.DataFrame()
        nodes = list(graph_parent.nodes)
        self.start_nodes = start_nodes(self)
        self.graph = graph_parent.subgraph(self.start_nodes).copy()
        self.vectors = sp.sparse.hstack([vectors_parent[:,nodes.index(n)]
                                         for n in self.start_nodes])
        self.n_seeds = n_seeds
        self.dct = dct
        self.point = point
        self.insert = insert
        self.delete = delete
        self.rvs = rvs
        self.create = create
        self.crossover = crossover
    
    def __str__(self):
        return f"Model\tparent: '{self.graph_parent.name}'\n" +\
               f"\tyear_start: {self.year_start}\n" +\
               f"\tstart_nodes: {self.start_nodes}\n" +\
               f"\tn_seeds: {self.n_seeds}\n" +\
               f"\tpoint: ({self.point[0]:.4f}, {self.point[1]:.4f})\n" +\
               f"\tinsert: ({self.insert[0]}, {self.insert[1]:.4f}, {type(self.insert[2])})\n" +\
               f"\tdelete: ({self.delete[0]}, {self.delete[1]:.4f})"
    
    def __repr__(self):
        return self.__str__()
    
    def evolve(self, until, record=False):
        """ Evolves a graph based on vector representations 
        until `until (lambda wiki.Model) == True`
        """
        year_start = self.year
        while not until(self):
            sys.stdout.write(f"\r{year_start} > {self.year} "+\
                             f"n={self.graph.number_of_nodes()}    ")
            sys.stdout.flush()
            self.initialize_seeds()
            self.mutate_seeds()
            self.create_nodes()
            if record:
                self.record = pd.concat([self.record] + \
                                        [pd.DataFrame({'Year': self.year,
                                                       'Parent': seed,
                                                       'Seed number': i,
                                                       'Seed vectors': seed_vec}, index=[0])
                                         for seed, seed_vecs in self.seeds.items()
                                         for i, seed_vec in enumerate(seed_vecs)],
                                        ignore_index=True,
                                        sort=False)
            self.year += 1
        print('')
    
    def initialize_seeds(self):
        nodes = list(self.graph.nodes)
        for i, node in enumerate(nodes):
            if node not in self.seeds.keys():
                self.seeds[node] = []
            if node not in self.thresholds.keys():
                self.thresholds[node] = []
            while len(self.seeds[node]) < self.n_seeds:
                self.seeds[node] += [self.vectors[:,i].copy()]
            while len(self.thresholds[node]) < self.n_seeds:
                self.thresholds[node] += [self.create(1)[0]]
    
    def mutate_seeds(self):
        for node, vecs in self.seeds.items():
            self.seeds[node] = [Model.mutate(vec, self.rvs, self.point,
                                             self.insert, self.delete)
                                for vec in vecs]
    
    def crossover_seeds(self):
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            seeds_i = sp.sparse.hstack(self.seeds[nodes[i]])
            for j in range(i+1,len(nodes)):
                seeds_j = sp.sparse.hstack(self.seeds[nodes[j]])
                similarity = smp.cosine_similarity(seeds_i.transpose(),
                                                   seeds_j.transpose())
                for k,l in np.argwhere(similarity>self.threshold):
                    cross = Model.crossover(seeds_i[:,k], seeds_j[:,l])
                    choice = np.random.choice(2)
                    self.seeds[nodes[i]][k] = cross if choice else self.vectors[:,i]
                    self.seeds[nodes[j]][l] = cross if not choice else self.vectors[:,j]
    
    def create_nodes(self):
        nodes = list(self.graph.nodes)
        for i, node in enumerate(nodes):
            parent = self.vectors[:,i]
            seeds = sp.sparse.hstack(self.seeds[node])
            sim_to_parent = smp.cosine_similarity(parent.transpose(), seeds.transpose())
            for j, seed_vec in enumerate(self.seeds[node]):
                if sim_to_parent[0,j] < self.thresholds[node][j]:
                    Model.connect(seed_vec, self.graph, self.vectors, self.dct)
                    self.vectors = sp.sparse.hstack([self.vectors, seed_vec])
                    self.seeds[node].pop(j)
                    self.thresholds[node].pop(j)
        for node in self.graph.nodes:
            if 'year' not in self.graph.nodes[node].keys():
                self.graph.nodes[node]['year'] = self.year
    
    @staticmethod
    def mutate(x, rvs, point=(0,0), insert=(0,0,None), delete=(0,0)):
        """ Mutates vector ``x`` with point mutations,
        insertions, and deletions. Insertions and point
        mutations draw from a random process ``rvs``.
        
        Parameters
        ----------
        x: spipy.sparse.csc_matrix
        rvs: lambda (n)-> float
            returns ``n`` random weights in [0,1]
        point: tuple (int n, float p)
            n = number of elements to insert
            p = probability of insertion for each trial
        insert: tuple (n, p, iterable s)
            s = set of elements from which to select
                if None, select from all zero elements
        delete: tuple (n, p)
        max_weight: float
        """
        data = x.data
        idx = x.indices
        if idx.size==0:
            return x
        n_point = np.random.binomial(point[0], point[1])
        i_point = np.random.choice(x.size, size=n_point, replace=False)
        data[i_point] = rvs(n_point)
        # insertion
        n_insert = np.random.binomial(insert[0], insert[1])
        for _ in range(n_insert):
            while True:
                insert_idx = np.random.choice(insert[2]) if insert[2]\
                    else np.random.choice(x.shape[0])
                if insert_idx not in idx: break
            idx = np.append(idx, insert_idx)
            data = np.append(data, rvs(1))
        # deletion
        n_delete = np.random.binomial(delete[0], delete[1])
        i_delete = np.random.choice(idx.size, size=n_delete, replace=False)
        idx = np.delete(idx, i_delete)
        data = np.delete(data, i_delete)
        y = sp.sparse.csc_matrix((data, (idx, np.zeros(idx.shape, dtype=int))),
                                 shape=x.shape)
        return y
    
    @staticmethod
    def connect(seed_vector, graph, vectors, dct, top_words=10, match_n=6):
        """
        
        Parameters
        ----------
        seed_vector: scipy.sparse.csc_matrix
        graph: networkx.DiGraph (not optional)
        vectors: scipy.sparse.csc_matrix (not optional)
        dct: gensim.corpora.dictionary (not optional)
        top_words: int (default=5)
        match_n: int
            how many words should be matched by...
        """
        seed_top_words, seed_top_idx = Model.find_top_words(seed_vector, dct)
        seed_name = ' '.join(seed_top_words)
        nodes = list(graph.nodes)
        graph.add_node(seed_name)
        for i, node in enumerate(nodes):
            node_vector = vectors[:,i]
            node_top_words, node_top_idx = Model.find_top_words(node_vector, dct)
            if len(set(seed_top_idx).intersection(set(node_vector.indices))) >= match_n or\
               len(set(node_top_idx).intersection(set(seed_vector.indices))) >= match_n:
                graph.add_edge(node, seed_name)
    
    @staticmethod
    def find_top_words(x, dct, top_n=10):
        """
        
        Parameters
        ----------
        x: scipy.sparse.csc_matrix
        dct: gensim.corpora.dictionary
        top_n: int

        Returns
        -------
        words:
        idx_vector: 
        """
        top_idx = np.argsort(x.data)[-top_n:]
        idx = [x.indices[i] for i in top_idx]
        words = [dct[i] for i in idx]
        words_nostop = gpp.remove_stopwords(' '.join(words)).split(' ')
        idx_keep = list(map(lambda x: words.index(x), set(words).intersection(words_nostop)))
        idx_nostop = list(map(idx.__getitem__, idx_keep))
        return words_nostop, idx_nostop
        
    @staticmethod
    def crossover(v1, v2):
        """ Crosses two vectors by combining half of one
        and half of the other.

        Parameters
        ----------
        v1, v2: scipy.sparse.matrix

        Returns
        -------
        v3: scipy.sparse.matrix
        """
        idx1 = np.random.choice(v1.size, size=int(v1.size/2))
        idx2 = np.random.choice(v2.size, size=int(v2.size/2))
        data = np.array([v1.data[i] for i in idx1] +
                        [v2.data[i] for i in idx2])
        idx = np.array([v1.indices[i] for i in idx1] +
                       [v2.indices[i] for i in idx2])
        v3 = sp.sparse.csc_matrix((data, (idx, np.zeros(idx.shape, dtype=int))),
                                  shape=v1.shape)
        return v3
