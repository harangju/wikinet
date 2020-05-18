import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'module'))
import wiki
import pickle, dill
# import datetime
import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp

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

path_dict = os.path.join()
path_results = os.path.join()
path_run = os.path.join()
save_models = False

print("Loading dictionary...")
dct = pickle.load(open(path_dict, 'rb'))

print("Loading network for topics...")
networks = {}
for topic in topics:
    print(f"\t'{topic}'")
    network = wiki.Net()
    network.load_graph(os.path.join(path, topic+'.pickle'))

print("Initializing model parameters...")
n_seeds = 2
n_models = 3
year_start = 0
start_condition = lambda m: [
    n for n in m.graph_parent.nodes
    if m.graph_parent.nodes[n]['year'] <= year_start
]
end_condition = lambda m:
    (len(m.graph.nodes) >= len(m.graph_parent.nodes)) or (m.year > 2200)
stats = pd.DataFrame()

print("Starting simulations...")
for topic, network in networks.items():
    print(topic)
    print('Analyzing priors...')
    tfidf = network.graph.graph['tfidf']
    yd = year_diffs(network.graph)
    md = word_diffs(network.graph, tfidf)
    a_md, b_md, r_md, p_md, stderr = sp.stats.linregress(np.abs(yd), md)
    swd = sum_abs_weight_differences(network.graph, tfidf)
    a_swd, b_swd, r_swd, p_swd, stderr = sp.stats.linregress(np.abs(yd), swd)
    rvs = lambda n: tfidf.data[np.random.choice(tfidf.data.size, size=n)]
    mu_sawd = np.mean(np.sum(np.abs(rvs((1,100000))-rvs((1,100000))), axis=0))
    nb = neighbor_similarity(network.graph, tfidf)
    mu_nb, std_nb = sp.stats.norm.fit(nb)
    p_point, p_insert, p_delete = a_swd/mu_sawd, a_md/2, a_md/2
    new_stats = pd.DataFrame(
        [[
            p_point, p_insert, p_delete, a_md, b_md, r_md, p_md,
            a_swd, b_swd, r_swd, p_swd, mu_sawd, mu_nb, std_nb
        ]],
        columns=[
            'p_pt', 'p_in', 'p_de',
            'a (man)', 'b (man)', 'r (man)', 'p (man)',
            'a (swd)', 'b (swd)', 'r (swd)', 'p (swd)',
            'mu (sawd)', 'mu (nei)', 'std (nei)'
        ]
    )
    display(HTML(new_stats.to_html()))
    stats = pd.concat([stats, new_stats], ignore_index=True)
    for i in range(n_models):
        print(f"Running model {i}...")
        model = wiki.Model(
            graph_parent=network.graph,
            vectors_parent=tfidf,
            year_start=year_start,
            start_nodes=start_condition,
            n_seeds=n_seeds,
            dct=dct,
            point=(1, p_point),
            insert=(1, p_insert, list(set(tfidf.indices))),
            delete=(1, p_delete),
            rvs=rvs,
            create=lambda n: np.random.normal(loc=mu_nb, scale=std_nb, size=n)
        )
        model.evolve(until=end_condition)
        if save_models:
            dill.dump(
                model,
                open(
                    os.path.join(base_dir, now, f"model_{topic}_{i}.pickle"),
                    'wb'
                )
            )
    print('')
pickle.dump(stats, open(os.path.join(base_dir, now, 'stats.pickle'), 'wb'))
