# Import helper functions
from preprocess import importQuery, importTweets, buildIndex, lengthOfDocument
import spacy
from nltk.corpus import wordnet
from results import retrieve
from write import resultFileCreation

nlp = spacy.load('en_core_web_lg')

# Importing the Tweets into documents and the queries
# Load the tweet list.
# {'34952194402811904': 'Save BBC World Service from Savage Cuts http://www.petitionbuzz.com/petitions/savews', ...}
documents = importTweets()
# Load the list of queries.
# {1: ['bbc', 'world', 'servic', 'staff', 'cut'], ...}
queries = importQuery()[1]

def getSim(word, syn):   #Finds the Similarity between the queries
    tokens = nlp(' '.join([word, syn]))
    simOne, simTwo = tokens[0], tokens[1]

    return simOne.similarity(simTwo)

def getSyns(queryList):   # Finds the synonyms for the query
    synonyms=[]
    for word in queryList:
        synsOfWord = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if l.name() not in synsOfWord:
                    synsOfWord.append(l.name())
        synonyms.append(synsOfWord)
    return synonyms

def canExpand(syn, queryList):  # Helper function for expanding the query
    count = 0
    thresh = 0.1
    for word in queryList:

        sim = getSim(word, syn)
        if sim >= thresh:
            count += 1
    return count >=2

def expand(query):   # Performs the query expansion
    newQuery = query.copy()
    syns = getSyns(query)
    for synWordList in syns:
        for synWord in synWordList:
            if canExpand(synWord, query) and synWord not in newQuery:
                newQuery.append(synWord)
    return newQuery

def main():   # Runs the main for the query expansion
    print("\n CSI 4107 - Neural Information Retrieval Using Query Expansion \n")
    print("\n Initializing The Query Expansion... \n")
    # Expands the query using the query expansion method
    newQueryDict = dict()
    for queryIndex in queries:
        print("Creating the queryExpansion:", queryIndex,'/', len(queries), " ", end = '\r' )
        expanded = expand(queries[queryIndex])
        newQueryDict[queryIndex] = expanded
    # Build the inverted index.
    index = buildIndex(documents)
    # Get the length of each document.
    document_length = lengthOfDocument(index, documents)
    print("\n Query Expansion Done!!! \n")
    print("\n Retrieval and Ranking... \n")
    # Get length of query.
    ranking = retrieve(newQueryDict, index, document_length)
    print("\n Retrieval and Ranking Done! \n")
    print("\n Starting to create Result File... \n")
    resultFileCreation(ranking)
    print("\n Result File Creation Done! \n")

if __name__ == "__main__":
    main()
