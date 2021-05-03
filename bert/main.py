# Library Import
from sentence_transformers import SentenceTransformer
from scipy import spatial
from tqdm import tqdm
from nltk.corpus import stopwords

# File Import
from preprocess import importTweets,importQuery
from write import resultFileCreation

def bert_main():
    # Tweet data using a Vectorizer
    tweets = importTweets(True)
    tweetsk = list(tweets.keys())
    tweetsl = list(tweets.values())

    # Query data using a Vectorizer
    queries = importQuery(True)
    queriesl = list(queries.values())

    # Embedding the Tweet data words using a bert model.
    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    print("-"*70)
    print("Embedding the tweet strings with the Bert token model...")
    tweet_embeddings = bert_model.encode(tweetsl, batch_size = 500, show_progress_bar = True)

    # Embedding the Query data words using a bert model.
    print("-"*70)
    print("Embedding the Query strings with the Bert token model...")
    query_embeddings = bert_model.encode(queriesl, batch_size = 500, show_progress_bar = True)

    # Calculation the Cosine Similarity for the embedded words.
    print("-"*70)
    print("Calculating the Cosine Similarity for the Bert embedded Tweets and Queries...")
    Rankings = {}
    for q in range(0,len(queriesl)):
        # Dictionary to sort the Cosine Similarity of each document per query.
        docCurrentQuery = {}
        for t in range(0,len(tweetsl)):
            docCurrentQuery[tweetsk[t]] = 1 - spatial.distance.cosine(tweet_embeddings[t], query_embeddings[q])
        # Sorting the document in descending order of the Cosine Similarity per query.
        docCurrentQuery = dict(sorted(docCurrentQuery.items(), key=lambda item: item[1],reverse=True))

        # Creating a new dictionary of only the Top 1000 documents for each query.
        doc_counter = 1
        docCurrentQuery_1000 = {}
        for key, value in docCurrentQuery.items():
            if(doc_counter <= 1000):
                docCurrentQuery_1000[key] = value
                doc_counter += 1
            else:
                break
        Rankings[q+1] = docCurrentQuery_1000

    print("-"*70)
    print("Creating a results file with all the required details...")
    # Creating a txt file with the results
    resultFileCreation(Rankings, True)
    print("-"*70)
    print("Results file is created (visit the dist folder)")
    print("-"*70)
    
if __name__ == "__main__":
    bert_main()