# Import numpy and torch.
import numpy as np
import torch
import nltk

from random import randint
from models import InferSent

# Import helper functions
from preprocess import importQuery, importTweets
from write import resultFileCreation

nltk.download('punkt')

USE_CUDA = False
V = 1
MODEL_PATH = 'encoder/infersent%s.pkl' % V
W2V_PATH = 'GloVe/glove.840B.300d.txt'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}


def cosine(u, v):
    '''
    Calculate the Cosine Similarity between two Vectors

    :param u, v: Vectors to be compared.
    :return: The Cosine Similarity between u and v.
    :rtype: float
    '''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def main():

    # Dictionary for Final Rankings.
    ranking = dict()

    print("\n CSI 4107 - Microblog information retrieval system \n")

    print("\n Importing Query Files and Documents... \n")

    # Load the tweet list.
    # {'34952194402811904': 'Save BBC World Service from Savage Cuts http://www.petitionbuzz.com/petitions/savews', ...}
    tweets_dict = importTweets()

    # Load the list of queries.
    # {1: ['bbc', 'world', 'servic', 'staff', 'cut'], ...}
    queries_dict = importQuery()

    print("\n Importing Done! \n")

    print("\n Initializing InferSent Model... \n")

    # Initialize InferSent Model.
    infersent = InferSent(params_model)

    # Load Infersent v1 Model Encoder.
    infersent.load_state_dict(torch.load(MODEL_PATH))

    # Use GPU Mode
    infersent = infersent.cuda() if USE_CUDA else infersent

    # Load Pre-trained GloVe Model.
    infersent.set_w2v_path(W2V_PATH)

    print("\n InferSent Initialization Done! \n")

    print("\n Building Vocabulary from Tweets... \n")

    # Deconstruct the dictionary of Documents to Document ID, and Document Contents.
    tweets = list(tweets_dict.values())
    tweet_ids = list(tweets_dict.keys())

    # Deconstruct the dictionary of Queries to Query Contents, since we can replicate Query ID.
    queries = list(queries_dict.values())

    # Build the Infersent Vocabulary based on all the Documents' Contents.
    infersent.build_vocab(tweets, tokenize=False)

    print("\n Vocabulary Completed! \n")

    print("\n Building Document & Query Vectors... \n")

    doc_embeddings = infersent.encode(tweets, bsize=128, tokenize=False, verbose=True)
    query_embeddings = infersent.encode(queries, bsize=128, tokenize=False, verbose=True)

    print("\n Building Document & Query Vectors Done! \n")

    print("\n Retrieval and Ranking... \n")

    dranking = dict()

    for query_id in range(len(queries)):
        print (dranking)
        # Encoded array starts at 0 for first chronological document.
        current_document = 0

        # Calculate the Cosine Similarity between the current Query, and corpus of Documents.
        for tweet_id in tweet_ids:
            # Calculate the Cossine Sim
            dranking[tweet_id] = cosine(
                doc_embeddings[current_document],
                query_embeddings[query_id]
            )
            current_document += 1

        # Put the ranking of Documents in Descending order into ranking.
        ranking[query_id + 1] = {k: v for k, v in sorted(dranking.items(), key=lambda dranking: dranking[1], reverse=True)[:1000]}

        # Create the resulting file.
        print ("Query " + str(query_id) + " Done.")
        dranking.clear()

    resultFileCreation(ranking)

    print("\n Retrieval and Ranking Done! \n")

if __name__ == "__main__":
    main()
