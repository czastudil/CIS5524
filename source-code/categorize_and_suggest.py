from sentence_transformers import SentenceTransformer, util
import torch
import random
import os
import pickle
from visualize import *
from utilities import *

# Get article corpus
def get_articles(category):
    category_map = get_category_map()
    article_map = read_article_map()
    nodes = category_map[category]
    articles = []
    for n in nodes:
        articles.append(article_map[n])
    return articles

# Get category corpus
def get_categories():
    CATEGORY_MAP_FILENAME = 'data/wiki-topcats-categories.txt'
    cats = []
    with open(CATEGORY_MAP_FILENAME) as file:
        for line in file:
            split = line.split(';')
            category = split[0].split(':')[1]
            cats.append(category)
    return cats

# Load the saved category suggestion model
model = SentenceTransformer('saved_model')

# Embed the category corpus if it's not cached, otherwise load it from disc
cat_embedding_cache_path = 'category-embeddings.pkl'
if not os.path.exists(cat_embedding_cache_path):
    cat_corpus_sentences = get_categories()
    random.shuffle(cat_corpus_sentences)
    print("Encode the category corpus. This might take a while")
    cat_corpus_embeddings = model.encode(cat_corpus_sentences, show_progress_bar=True, convert_to_tensor=True)
    print("Store file on disc")
    with open(cat_embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': cat_corpus_sentences, 'embeddings': cat_corpus_embeddings}, fOut)
else:
    print("Load pre-computed category embeddings from disc")
    with open(cat_embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        cat_corpus_sentences = cache_data['sentences']
        cat_corpus_embeddings = cache_data['embeddings']

# TODO: Hard-coded query, should be able to input to program
query = "Usability"
# Suggest the top 5 closet categories
k = 5
# Embed the query for the category model
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute the cosine similarity of the query to all of the category corpus
cos_scores = util.cos_sim(query_embedding, cat_corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=k)
print("\n\n======================\n\n")
print("Query:", query)
print("\nTop 5 most similar categories in corpus:")

top_cats = []
for score, idx in zip(top_results[0], top_results[1]):
    top_cats.append(cat_corpus_sentences[idx])
    print(cat_corpus_sentences[idx], "(Score: {:.4f})".format(score))

print('Top Result:', top_cats[0])

# Load the saved article suggestion model
model2 = SentenceTransformer('saved_model_articles')

# Embed the article corpus if it's not cached, otherwise load it from disc
article_embedding_cache_path = 'article-embeddings.pkl'
article_corpus_sentences = get_articles(top_cats[0])
random.shuffle(article_corpus_sentences)
article_corpus_embeddings = model2.encode(article_corpus_sentences, show_progress_bar=True, convert_to_tensor=True)
# Embed the query for the article model
query_embedding_2 = model2.encode(query, convert_to_tensor=True)
# Suggest the top 15 closest articles
k=15
# Compute the cosine similarity of the query to all of the article corpus
cos_scores_article = util.cos_sim(query_embedding_2, article_corpus_embeddings)[0]
top_results_article = torch.topk(cos_scores_article, k=k)

print("\n\n======================\n\n")
print("Query:", query)
print("\nTop 15 most similar articles in corpus:")
top_articles = []
for score, idx in zip(top_results_article[0], top_results_article[1]):
    top_articles.append(article_corpus_sentences[idx])
    print(article_corpus_sentences[idx], "(Score: {:.4f})".format(score))

# Visualize the ego network 
draw_network(query, top_articles, top_cats[0])