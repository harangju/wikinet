import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'module'))
import pyximport; pyximport.install()
import wiki
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import wikihelper as wh
import scipy.stats as ss

topics = ['earth science']
base_dir = '/Users/harangju/Developer/data/wiki/'
dct = pickle.load(open(base_dir + 'models/dict.model','rb'))

print('Loading networks...')
networks = {}
for topic in topics:
    networks[topic] = wiki.Net()
    networks[topic].load_graph(base_dir + 'graphs/dated/' + topic + '.pickle')

for topic, network in networks.items():
    print(f"\nTopic: {topic}")
    graph = networks[topic].graph
    tfidf = graph.graph['tfidf']
    print('\tMeasuring priors...')
    yd = wh.year_diffs(graph)
    _, _, r_sw, _, _ = ss.linregress(np.abs(yd), wh.sum_weight_diffs(graph, tfidf))
    mu_wd, _ = ss.norm.fit(wh.weight_diffs(graph, tfidf))
    mu_weight = np.mean(tfidf.data)
    _, _, r_wd, _, _ = ss.linregress(np.abs(yd), wh.word_diffs(graph, tfidf))
    mu_ns, std_ns = ss.norm.fit(wh.neighbor_sim(graph, tfidf))
    print('\tInitializing...')
    model = wiki.Model(graph_parent=graph,
                       vectors_parent=tfidf,
                       year_start=-500,
                       n_seeds=2,
                       dct=dct,
                       point=(r_sw/mu_wd, mu_weight/r_sw),
                       insert=(1, r_wd/2, list(set(tfidf.indices))),
                       delete=(1, r_wd/2),
                       rvs=lambda n: tfidf.data[np.random.choice(tfidf.data.size, size=n)],
                       create=lambda n: np.random.normal(loc=mu_ns, scale=std_ns, size=n),
                       crossover=mu_ns+3*std_ns)
    print('\tRunning model...')
    model.evolve(until=-450)
    print('\n\tAnalyzing...')
    nodes = list(model.graph.nodes)
    model.record['Similarity (parent)'] = [wh.cos_sim(model.record.iloc[i]['Seed vectors'], 
                                                      model.vectors[:,nodes.index(
                                                          model.record.iloc[i]['Parent'])])
                                           for i in range(len(model.record.index))]
    print('\tSaving...')
    model.rvs = None; model.create = None
    pickle.dump(model, open(f"models/model_{topic}_{0}.p", 'wb'))
