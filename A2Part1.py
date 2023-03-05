from sentence_transformers import SentenceTransformer
from numpy import dot
from math import sqrt
import json
from sklearn import metrics
from numpy import dot
from numpy.linalg import norm

def get_tweets():
    tweets = [] #used to be tweets = None - do I need to leave that or can I change it to this? Because it doesn't work if I leave it to None
    #data = []
    temp = {}
    with open('tweets-utf-8.json') as file:
        for line in file:
            temp = json.loads(line)
            tweets.append(temp['text'])
    return tweets

def sort_by_sim(query_embedding,document_embeddings,documents):
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = {}
    similarity = 0
    for embedding, doc in zip(document_embeddings, documents):
        # cos_sim = metrics.pairwise.cosine_similarity(query_embedding, embedding)
        for a,b in zip(query_embedding,embedding):
            cos_sim = dot(a, b)/(norm(a)*norm(b))
        similarities.update({cos_sim: doc})
    return similarities #
    
def glove_top25(query,documents):
    return []

def minilm_top25(query,documents):
    return []
        
## Test Code

tweets = get_tweets()

for x in range(5):
    print(tweets[x])


print("**************GLOVE*****************")
for p in glove_top25("I am looking for a job.",tweets): print(p)

print("**************MINILM*****************")
for p in minilm_top25("I am looking for a job.",tweets): print(p)