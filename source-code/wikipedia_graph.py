import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

class WikipediaNetwork:
    
    def __init__(self):
        network_cache_path = 'wikipedia-network.txt'
        # If the network is not cached, create & cache it
        if not os.path.exists(network_cache_path):
            print("Creating the network...")
            ADJ_LIST_FILENAME = 'data/wiki-topcats.txt'
            self.g = nx.read_adjlist(ADJ_LIST_FILENAME, create_using=nx.DiGraph)
            print("Storing file on disc...")
            with open(network_cache_path, "wb") as fOut:
                pickle.dump(self.g, fOut)
        # Otherwise, load the network from the cache
        else:
            print("Loading saved graph from path")
            with open(network_cache_path, "rb") as fIn:
                self.g = pickle.load(fIn)
        print("Mapping articles...")
        self.article_mapping, self.node_mapping = self.read_article_map()
        nx.set_node_attributes(self.g, self.node_mapping, name='article_name')
        print("Mapping categories...")
        self.category_mapping, self.node_category_mapping = self.read_category_map()
        nx.set_node_attributes(self.g, self.node_category_mapping, name='categories')

    # Process the file containing the node number to article name mapping
    def read_article_map(self):
        ARTICLE_MAP_FILENAME = 'data/wiki-topcats-page-names.txt'
        article_map = dict()
        node_map = dict()
        with open(ARTICLE_MAP_FILENAME) as file:
            for line in file:
                split = line.split(' ', maxsplit=1)
                article_map[split[1].strip()] = split[0]
                node_map[split[0]] = split[1].strip()
        return article_map, node_map

    # Process the file containing category names and the articles within them (in numerical form)
    def read_category_map(self):
        CATEGORY_MAP_FILENAME = 'data/wiki-topcats-categories.txt'
        cat_map = dict()
        node_cat_map = dict()
        with open(CATEGORY_MAP_FILENAME) as file:
            for line in file:
                split = line.split(';')
                nodes = split[1].split()
                category = split[0].split(':')[1]
                cat_map[category] = nodes
                for n in nodes:
                    if n in node_cat_map:
                        node_cat_map[n].append(category)
                    else:
                        node_cat_map[n] = [category]
        return cat_map, node_cat_map

    def visualize_category(self, category_name):
        try:
            cats = self.category_mapping[category_name]
            g = nx.Graph()
            for a in cats:
                g.add_edge(self.node_mapping[a], category_name)
            plt.figure(figsize=(20,15))
            pos_nodes = nx.circular_layout(g)
            pos_nodes[category_name] = np.array([0, 0])
            nx.draw_networkx(g, pos=pos_nodes, node_shape="s",  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
            plt.show()
        except:
            print('Category does not exist.')

    def visualize_article_network(self, article_name):
        print("Visualizing article...")
        try:
            article_num = self.article_mapping[article_name]
            print("Create subgraph")
            nodes = list(self.g.neighbors(article_num))
            nodes.append(article_num)
            print(nodes)
            subgraph = self.g.subgraph(nodes).copy()
            print("Subgraph created")
            plt.figure(figsize=(20,15))
            pos_nodes = nx.circular_layout(subgraph)
            pos_nodes[article_num] = np.array([0, 0])
            print("Get node attributes")
            labels = nx.get_node_attributes(subgraph, name='article_name')
            print("Draw network")
            nx.draw_networkx(subgraph, pos=pos_nodes, labels=labels, node_shape="s", node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
            pos_attrs = {}
            for node, coords in pos_nodes.items():
                pos_attrs[node] = (coords[0], coords[1] + 0.08)
            nx.draw_networkx_labels(subgraph, pos_attrs, labels=nx.get_node_attributes(subgraph, 'categories'))
            plt.show()
        except:
           # print('Article does not exist.')
    
    # Utility function
    def find_article_hubs(self):
        deg_map = dict()
        for n in list(self.g.nodes):
            deg_map[n] = self.g.degree(n)
        sorted_deg = sorted(deg_map.items(), key=lambda x:x[1], reverse=True)
        return sorted_deg

    # Utility function
    def find_category_hubs(self):
        article_count = dict()
        for cat in self.category_mapping.keys():
            article_count[cat] = len(self.category_mapping[cat])
        sorted_deg = sorted(article_count.items(), key=lambda x:x[1], reverse=True)
        return sorted_deg
    
    # Utility function
    def find_category_overlap(self, category):
        categories = set()
        for n in self.category_mapping[category]:
            for c in self.category_mapping.keys():
                if n in c:
                    categories.add(c)
        return categories
    
    # Utility function
    def find_smallest_deg(self):
        deg_map = dict(self.g.degree())
        sorted_deg = sorted(deg_map.items(), key=lambda x:x[1], reverse=False)
        return sorted_deg
    
    # Known from SNAP dataset - size, edges, nodes in largest WCC,
    # edges in largest WCC, nodes in largest SCC, edges in largest SCC,
    # average clustering coefficient, number of triangles,
    # fraction of closed triangles, diameter, 90th percentile effective diameter
    def get_network_metrics(self, category = None):
        # Calculate the average degree
        deg_sum = 0
        for n in list(self.g.nodes):
            deg_sum += self.g.degree(n)
        avg_deg = deg_sum / len(list(self.g.nodes))
        # Determine article hubs
        article_hubs = self.find_article_hubs()[:5]
        # Useful for finding articles for visualization
        smallest_deg = self.find_smallest_deg()
        # Determine category hubs
        cat_hubs = self.find_category_hubs()[:3]
        # Find overlapping categories for a specific category if passed the parameter
        overlap = None
        if category != None:
            overlap = self.find_category_overlap('Living_people')
        return overlap, avg_deg, article_hubs, smallest_deg, cat_hubs