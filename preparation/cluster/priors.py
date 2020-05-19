import os, sys
sys.path.insert(1, sys.path[0])
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import sklearn.preprocessing as skp
import sklearn.metrics.pairwise as smp
import cython_manhattan as cm

def year_diffs(graph):
    return [graph.nodes[node]['year'] - graph.nodes[neighbor]['year']
            for node in graph.nodes
            for neighbor in list(graph.successors(node))]

def neighbor_similarity(graph, tfidf):
    nodes = list(graph.nodes)
    return [smp.cosine_similarity(tfidf[:,nodes.index(node)].transpose(),
                                  tfidf[:,nodes.index(neighbor)].transpose())[0,0]
            for node in nodes
            for neighbor in list(graph.successors(node))]

def sparse_manhattan(X,Y=None):
    X, Y = smp.check_pairwise_arrays(X, Y)
    X = sp.sparse.csr_matrix(X, copy=False)
    Y = sp.sparse.csr_matrix(Y, copy=False)
    res = np.empty(shape=(X.shape[0],Y.shape[0]))
    cm.cython_manhattan(
        X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, res
    )
    return res

def word_diffs(graph, tfidf):
    dists = sparse_manhattan(X=skp.binarize(tfidf).transpose())
    nodes = list(graph.nodes)
    return [dists[nodes.index(node), nodes.index(neighbor)]
            for node in nodes
            for neighbor in list(graph.successors(node))]

def sum_abs_weight_differences(graph, tfidf):
    nodes = list(graph.nodes)
    diff = []
    for node in nodes:
        for neighbor in graph.successors(node):
            v1 = tfidf[:,nodes.index(node)]
            v2 = tfidf[:,nodes.index(neighbor)]
            idx = np.concatenate([v1.indices, v2.indices])
            diff.append( np.sum(np.absolute(v1[idx]-v2[idx])) )
    return diff

def sum_weight_differences(graph, tfidf):
    nodes = list(graph.nodes)
    diff = []
    for node in nodes:
        for neighbor in graph.successors(node):
            v1 = tfidf[:,nodes.index(node)]
            v2 = tfidf[:,nodes.index(neighbor)]
            idx = np.concatenate([v1.indices, v2.indices])
            diff.append( np.sum(v1[idx]-v2[idx]) )
    return diff

def bin_distribution(data, steps=30, scale='log'):
    if scale=='log':
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), steps)
    elif scale=='linear':
        bins = np.linspace(np.min(data), np.max(data), num=steps)
    hist, edges = np.histogram(data, bins=bins)
    return hist, edges, bins

def plot_distribution(data):
    hist, edges, bins = bin_distribution(data)
#     hist_norm = hist/(bins[1:] - bins[:-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bins[:-1],
                             y=hist/len(data),
                             mode='markers'))
    fig.update_layout(template='plotly_white',
                      xaxis={'type': 'log',
                             'title': 'x'},
                      yaxis={'type': 'log',
                             'title': 'P(x)'})
    fig.show()
    return fig
