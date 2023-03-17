import numpy
from sentence_transformers import SentenceTransformer, util
from numpy import dot
from math import sqrt
import json
from sklearn import metrics
from numpy import dot
from numpy.linalg import norm
from array import array
import numpy
from operator import itemgetter

import torch

def get_tweets():
    tweets = []
    temp = {}
    with open('tweets-utf-8.json') as file:
        for line in file:
            temp = json.loads(line)
            tweets.append(temp['text'])
    return tweets

def sort_by_sim(query_embedding,document_embeddings,documents):
    similarities = []

    for d_embedding, document in zip(document_embeddings, documents):
        cos_sim = metrics.pairwise.cosine_similarity(query_embedding.reshape(1,-1), d_embedding.reshape(1,-1))
        similarity = numpy.asarray(cos_sim)
        similarity_item = similarity.item()
        similarities.append((similarity_item, document))
    
    similarities.sort(key=lambda x:x[0], reverse=True)
    # sorted(similarities, key=itemgetter(0), reverse=True)

    return similarities
    
def glove_top25(query,documents):
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.840B.300d')
    
    query_embedding = model.encode(query)
    embeddings = model.encode(documents)
    similarity = []
    similarity = sort_by_sim(query_embedding, embeddings, documents)
    
    return similarity[0:25]

def minilm_top25(query,documents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)

    similarity = []
    similarity = sort_by_sim(query_embedding, doc_embeddings, documents)
    return similarity[0:25]
        
## Test Code

tweets = get_tweets()

print("**************GLOVE*****************")
for p in glove_top25("I am looking for a job.",tweets): print(p)

print("**************MINILM*****************")
for p in minilm_top25("I am looking for a job.",tweets): print(p)