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

def round10(x):
    return int(round(x / 10.0)) * 10

def networkx_to_igraph(nx_graph, vertex_id=None):
    nodes = list(nx_graph.nodes)
    ig_graph = ig.Graph()
    ig_graph.add_vertices(list(range(len(nodes))))
    ig_graph.vs['name'] = nodes
    ig_graph.vs['year'] = [nx_graph.nodes[n]['year'] for n in nodes]
    ig_graph.add_edges([
        (nodes.index(s), nodes.index(t)) for s,t in nx_graph.edges
    ])
    ig_graph.es['weight'] = [nx_graph.edges[s,t]['weight'] for s,t in nx_graph.edges]
    if vertex_id:
        ig_graph.vs['id'] = vertex_id
    return ig_graph

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
path_sim = os.path.join(path_base, 'communities', now)

print("Checking directory...")
if not os.path.isdir(path_sim):
    os.mkdir(path_sim)

print("Loading network for topics...")
networks = {}
for topic in [topics[index]]:
    print(f"\t'{topic}'", end=' ')
    networks[topic] = wiki.Net()
    networks[topic].load_graph(os.path.join(path_networks, topic+'.pickle'))
    years = sorted(
        nx.get_node_attributes(networks[topic].graph, 'year').values(),
        reverse=True
    )
    for n in networks[topic].graph.nodes:
        networks[topic].graph.nodes[n]['year'] = round10(networks[topic].graph.nodes[n]['year'])

print('')

Cjrs = 0.01

print("Detecting communities...")
print(f"Cjrs = {Cjrs}")
memberships = {}
improvements = {}
for topic in [topics[index]]:
    graph = networks[topic].graph
    nodes = list(graph.nodes)
    years = sorted(set(nx.get_node_attributes(graph, 'year').values()))
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
        n_iterations=-1
    )
    pickle.dump(
        (memberships[topic], improvements[topic]),
        open(os.path.join(path_sim, f"membership_{topic}.pickle"), 'wb')
    )
print('Done.')
