def read_article_map():
    ARTICLE_MAP_FILENAME = 'data/wiki-topcats-page-names.txt'
    article_map = dict()
    file = open(ARTICLE_MAP_FILENAME, "r")
    # For every node, map the node number to its corresponding name
    for line in file:
        split = line.split(' ', maxsplit=1)
        article_map[split[0]] = split[1].strip()
    return article_map

def read_adj_list():
    ADJ_LIST_FILENAME = 'data/wiki-topcats.txt'
    articles = []
    # Get the mapping of node numbers to their article names
    article_map = read_article_map()
    file = open(ADJ_LIST_FILENAME, "r")
    # For every node pair, create an entry of the two article names in the articles list
    for line in file:
        split = line.split(' ', maxsplit=1)
        articles.append((article_map[split[0].strip()], article_map[split[1].strip()]))
    return articles

def get_category_map():
    CATEGORY_MAP_FILENAME = 'data/wiki-topcats-categories.txt'
    cat_map = dict()
    with open(CATEGORY_MAP_FILENAME) as file:
        for line in file:
            split = line.split(';')
            nodes = split[1].split()
            category = split[0].split(':')[1]
            cat_map[category] = nodes
    return cat_map