import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_network(article, links, category):
    g = nx.DiGraph()
    for l in links:
        g.add_edge(article, l)
    fig, ax = plt.subplots()
    fig.figsize=(20,15)
    pos_nodes = nx.circular_layout(g)
    pos_nodes[article] = np.array([0, 0])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, category, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    nx.draw_networkx(g, pos=pos_nodes, node_shape="s",  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
    plt.show()

