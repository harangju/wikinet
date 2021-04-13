import networkx as nx

__all__ = ['GraphContainer']

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
                self._numbered.add_node(
                    self.nodes.index(node),
                    year = self.years.index(self.graph.nodes[node]['year'])
                )
                self._numbered.add_edges_from(
                    [
                        (self.nodes.index(node), self.nodes.index(succ))
                        for succ in self.graph.successors(node)
                    ]
                )
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
            self._years = [
                self.graph.nodes[n]['year']
                for n in self.graph.nodes()
            ]
            self._years = sorted(list(set(self._years)))
            return self._years

    @property
    def nodes_for_year(self):
        if self._nodes_for_year:
            return self._nodes_for_year
        else:
            self._nodes_for_year = {
                year: [
                    list(self.graph.nodes).index(n)
                    for n in self.graph.nodes
                    if self.graph.nodes[n]['year']==year
                ]
                for year in self.years
            }
            return self._nodes_for_year
