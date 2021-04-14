import sys
import pandas as pd
import dionysus as d
import networkx as nx

from .graph_container import GraphContainer

__all__ = ['PersistentHomology']

class PersistentHomology(GraphContainer):
    """``Net`` is a child of ``PersistentHomology``. So you can call any of the following with any ``wikinet.Net`` object.

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
                self.filtration, self.persistence, self.graph, self.nodes
            )
            return self._barcodes

    @staticmethod
    def compute_barcodes(f, m, graph, names):
        """Uses dionysus filtration & persistence (in reduced matrix form) to compute barcodes.

        f: dionysus.Filtration
            filtration
        m: dionysus.ReducedMatrix
            (see homology_persistence)
        names: list of strings
            names of node indices

        :return: pandas.DataFrame
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
            birth_nodes = [
                n for n in birth_simplex
                if graph.nodes[n]['year']==birth_year
            ]
            if m.pair(i) != m.unpaired:
                death_year = int(f[m.pair(i)].data)
                death_simplex = [names[s] for s in f[m.pair(i)]]
                death_nodes = [
                    n for n in death_simplex
                    if graph.nodes[n]['year']==death_year
                ]
            else:
                death_year = np.inf
                death_simplex = []
                death_nodes = []
            pair = m.pair(i) if m.pair(i) != m.unpaired else np.inf
            chain = m[pair] if pair != np.inf else m[i]
            simp_comp = [f[entry.index] for entry in chain]
            nodes = [
                node_list[idx] for simplex in simp_comp for idx in simplex
            ]
            barcodes.append(
                [
                    dim, birth_year, death_year, death_year-birth_year,
                    birth_simplex, death_simplex,
                    birth_nodes, death_nodes, list(set(nodes))
                ]
            )
        print('')
        barcodes.sort(key=lambda x: x[0])
        bar_data = pd.DataFrame(
            data=barcodes,
            columns=[
                'dim', 'birth', 'death', 'lifetime',
                'birth simplex', 'death simplex',
                'birth nodes', 'death nodes', 'homology nodes'
            ]
        )
        return bar_data
