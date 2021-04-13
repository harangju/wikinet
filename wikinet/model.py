import sys
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import gensim.parsing.preprocessing as gpp
import sklearn.metrics.pairwise as smp

from .persistent_homology import PersistentHomology

__all__ = ['Model']

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
        self.vectors = sp.sparse.hstack([
            vectors_parent[:,nodes.index(n)]
            for n in self.start_nodes
        ])
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
                self.record = pd.concat(
                    [self.record] + \
                    [
                        pd.DataFrame(
                            {
                                'Year': self.year,
                                'Parent': seed,
                                'Seed number': i,
                                'Seed vectors': seed_vec
                            },
                            index=[0]
                        )
                        for seed, seed_vecs in self.seeds.items()
                        for i, seed_vec in enumerate(seed_vecs)
                    ],
                    ignore_index=True, sort=False
                )
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
            self.seeds[node] = [
                Model.mutate(
                    vec, self.rvs, self.point, self.insert, self.delete
                )
                for vec in vecs
            ]

    def crossover_seeds(self):
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            seeds_i = sp.sparse.hstack(self.seeds[nodes[i]])
            for j in range(i+1,len(nodes)):
                seeds_j = sp.sparse.hstack(self.seeds[nodes[j]])
                similarity = smp.cosine_similarity(
                    seeds_i.transpose(), seeds_j.transpose()
                )
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
        y = sp.sparse.csc_matrix(
            (data, (idx, np.zeros(idx.shape, dtype=int))),
            shape=x.shape
        )
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
        data = np.array(
            [v1.data[i] for i in idx1] + [v2.data[i] for i in idx2]
        )
        idx = np.array(
            [v1.indices[i] for i in idx1] + [v2.indices[i] for i in idx2]
        )
        v3 = sp.sparse.csc_matrix(
            (data, (idx, np.zeros(idx.shape, dtype=int))), shape=v1.shape
        )
        return v3
