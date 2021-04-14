import sys
import bct
import random
import pickle
import numpy as np
import scipy as sp
import networkx as nx
import cpnet as cp
import gensim.utils as gu
import gensim.matutils as gmat

from .dump import Dump
from .persistent_homology import PersistentHomology

__all__ = ['Net']

class Net(PersistentHomology):
    """``Net`` is a wrapper for ``networkx.DiGraph``.
    Uses ``dionysus`` for persistence homology.

    tfidf: scipy.sparse.csc.csc_matrix
        sparse column matrix of tfidfs,
        ordered by nodes, also stored in
        ``self.graph.graph['tfidf']``, lazy
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
        """ Builds ``self.graph`` (``networkx.Graph``) from nodes (``list`` of ``string``). Set ``model`` (from ``gensim``) and ``dct`` (``gensim.corpora.Dictionary``) for weighted edges. Set ``filter_top`` to ``True`` only if you want the top "lead" section of the article.
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

        graph: networkx.DiGraph
            graph
        full_parents: bool
            whether to fill empty nodes that have
            all parents with non-empty 'year'

        :return: whether at least 1 empty node was filled
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

        graph: networkx.DiGraph
            graph
        dump: wiki.Dump
            dump
        queue: list of **strings**
            names of Wikipedia pages
        depth_goal: int
            depth goal
        nodes: list of **strings**
            names of Wikipedia pages
        filter_top: bool
            ``True`` for ``dump.load_page()``
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

        nodes: list of nodes
        dump: wiki.Dump
        model: gensim.modes.tfidfmodel.TfidfModel
        dct: gensim.corpora.Dictionary

        :rtype: scipy.sparse.csc.csc_matrix
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
