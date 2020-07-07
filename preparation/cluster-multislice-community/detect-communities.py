import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'module'))
import wiki
import pickle, dill
import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp
import leidenalg as la
import igraph as ig

exec(open('priors.py').read())

topics = [
    'anatomy', 'biochemistry', 'cognitive science', 'evolutionary biology',
    'genetics', 'immunology', 'molecular biology', 'chemistry', 'biophysics',
    'energy', 'optics', 'earth science', 'geology', 'meteorology',
    'philosophy of language', 'philosophy of law', 'philosophy of mind',
    'philosophy of science', 'economics', 'accounting', 'education',
    'linguistics', 'law', 'psychology', 'sociology', 'electronics',
    'software engineering', 'robotics',
    'calculus', 'geometry', 'abstract algebra',
    'Boolean algebra', 'commutative algebra', 'group theory', 'linear algebra',
    'number theory', 'dynamical systems and differential equations'
]

path_base = os.path.join('/cbica','home','harang','developer','data','wiki')
path_networks = os.path.join(path_base, 'dated')
path_sim = os.path.join(path_base, 'simulations', now)
save_models = True

print("Loading network for topics...")
networks = {}
for topic in [topics[index]]:
    print(f"\t'{topic}'", end=' ')
    networks[topic] = wiki.Net()
    networks[topic].load_graph(os.path.join(path_networks, topic+'.pickle'))
print('')

print("Checking directory...")
if not os.path.isdir(path_sim):
    os.mkdir(path_sim)

_topic = topics[index]
_networks = {_topic: networks[_topic]}

print("Detecting communities...")
Cjrs = 0.1
print(f"Cjrs = {Cjrs}")
memberships = {}
improvements = {}
for topic in [_topic]:
    print(f"Topic '{topic}'")
    fig = go.Figure()
    graph = networks[topic].graph
    nodes = list(graph.nodes)
#     sorted_nodes = sorted(
#         nodes,
# #         key=lambda item: membership[-1][nodes_by_year[-1].index(item)]
#         key=lambda node: graph.nodes[node]['year']
#     )
    years = sorted(nx.get_node_attributes(graph, 'year').values())
    nodes_by_year = [
        [n for n in nodes if graph.nodes[n]['year']<=year]
        for year in years
    ]
    memberships[topic], improvements[topic] = la.find_partition_temporal(
        [
            networkx_to_igraph(
                nx.subgraph(graph, nodes_by_year[i]),
                [nodes.index(n) for n in nodes_by_year[i]]
            )
            for i, year in enumerate(years)
        ],
        la.ModularityVertexPartition,
        interslice_weight=Cjrs,
    )
    pickle.dump(
        (memberships[topic], improvements[topic]),
        open(os.path.join(path_sim, f"membership_{topic}.pickle"), 'wb')
    )
