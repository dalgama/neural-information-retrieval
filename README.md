# CSI4107 Assignment 2 - Neural Information Retrieval System

## Reference

https://www.site.uottawa.ca/~diana/csi4107/A2_2021/A2_2021.htm

## Group Members

**Dmitry Kutin - 300015920**
**Dilanga Algama - 8253677**
**Joshua O Erivwo - 8887065**

## Task Distribution

Dmitry Kutin
- IR Model reused from Assignment 1, using a *InferSent* (sent2vec) implementation.
- Implementation & helper functions in the `infersent/` directory.
- Evaluation script for ease of use across our IR systems.
- README.md sections for InferSent (Approach 3), and Final Discussion for InferSent results.

Dilanga Algama
- IR Model recreated from Assignment 1, using a *BERT* model (SentenceTransformer) implementation.
- Implementation & helper functions in the `bert/` directory.
- README.md sections for BERT (Approach 1).

Joshua O Erivwo
- IR Model reused from Assignment 1, using query expansion techniques to create a new set of queries containing synonyms words
- Implementation & helper functions in the `query-expansion/` directory.
- README.md sections for Query Expansion (Approach 2).

## Neural Retrieval Methods

### BERT (Approach 1 - Dilanga Algama)

#### Functionality

Our task for this assignment was to implement improved versions of the Information Retrieval (IR) system we created in Assignment 2, for a collection of documents (Twitter messages). A quick recap of what our BERT code does as a whole is as follows:

We import both the data files, one with the test queries and the other with the list of tweets to format the information in a Python code readable manner and to organize the words for our functions to read (we used dictionaries to store the data). This step also runs the words through a stemming and stop word removal process that stems all the words and handles the removal of stop words from the list of words.

We use a BERT model to embed the Query and Document words (tweets) within each sentence. This step is done in batches to soften the load on the CPU when running the whole data set, especially for the 40,000+ documents.

We calculate the CosSim similarity using the Cosine similarity calculator from the `Scipy` dictionary to calculate the CosSim for the tweets, to find their similarity score to the query. We order the tweets in a dictionary from highest to lowest similarity score and pass this information to step 4.

We organize the data created from step 3 and write the information of the top 1000 for each query in the right order to a `.txt` file (`dist/bert/bert_results.txt`).

#### Discussion (Compared to Assignment 1)

Our BERT program did not perform as well as our results in Assignment 1, scoring a `P@10` TREC score of `0.1769` in comparison to `0.1833` which was the score we got for Assignment 1. The `MAP` value for Assignment 1 was `0.1679` and the BERT approach MAP value is `0.0880`. We believe this makes sense due to how BERT calculates the similarity of the different sentences. Firstly, BERT's evaluation changes based on the sequence or order of the words in the sentence since it looks like the words beside the word, on the left and right during the evaluation, whereas in Assignment 1 the order didn't matter. As long as the document had all the words the query had, it was considered 100% similar, especially when calculation the cosine similarity. This is the biggest reason we came to based on the final results.

Furthermore, there is always a possibility that the model misinterpreted the words within the sentences as other developers have mentioned when using the BERT model. Where the BERT approach may result in good scores but its sense of the answer or understanding in a commonsense scenario is not the same as a person would have when creating the whole picture for the system.

#### Algorithm and Data structure

Our implementation of the information retrieval system was based on the guidelines provided in assignment 2. The bert folder contains four python files containing the functions used in implementing BERT Neural IR system.

#### Project Specific Files

##### `main.py`:
In the `main()` function, we started by importing the important functions that were used for implementing the IR system. The first step was to import the tweets and the queries from the `assert folder`. By importing the tweets and queries from the `asset folder`, `step1: preprocessing` was being done using the `filterSentence` that was implemented directly in the `import` function. After importing the tweets and queries from the text and then filtering them.
Next, 'step2: word embedding' is done where we embed the words from the queries and documents with the BERT model. Once the embedding is done we calculate the Cosine Similarity scores for the word embedded Documents and Queries. To understand what was happening in the `main()` function, we created a set of print statements that would notify the user when the preprocessing and the embedding of the document and queries are done. The user then gets informed of the creation of the BERT result file. 

##### `preprocess.py`:
 This file contains the process of developing `step1:Preprocessing` using python. Below are the functions implemented in the `preprocess.py`

 - importTweets(bertMode = False): imports the tweets from the collection. We first started by opening the text files, then we filter the file using our filterSentence function. The bertMode variable is for then the filterSentence function is called.

 - importQuery(bertMode = False): imports query from the collection. Same process as the importTweet(). The bertMode variable is for then the filterSentence function is called.

 - filterSentence(sen, bertMode = False): Filters sentences from tweets and queries. This function builds a list of `stopwords` and then `tokenizes` each word in the sentences by removing any numerics, links, single characters, punctuation, extra spaces, or stopwords contained in the list. Each imported tweet and query runs through the `NLTK's stopword list`, our `custom stopword list` that included the `URLs and abbreviations`, and the provided `stopword list`. After this step, each word is tokenized and stemmed with `Porter stemmer`. Under the `additional libraries` section, we discussed in-depth the use of `tokenization`, `stopwords`, and `porter stemmer`. If this function is in bertMode a string is returned with all the remaining words, otherwise, a tokenized list of all the remaining words is returned.

 - listToString(list): Converts a list of words into a string.

##### `write.py`:
  This file contains the procedure for implementing `step4`. The function creates a table for each of the results generated in the `bert_results.py` and then stores it in the `dist/bert folder` as a text file.

#### Additional Libraries

##### Prettytable (`prettytable.py`):  
A helper library to format the output for the `Results.txt` file. Used in the implementation of the `write.py`.

#### Results for Query 3 & 20

##### Query 3

```
Topic_id  Q0  docno              rank  score                tag 
3         Q0  34410414846517248  1     0.9017896056175232   myRun 
3         Q0  35088534306033665  2     0.9016108512878418   myRun 
3         Q0  32910196598636545  3     0.8939297199249268   myRun 
3         Q0  35032969643175936  4     0.8846487402915955   myRun 
3         Q0  34728356083666945  5     0.8838127851486206   myRun 
3         Q0  33254598118473728  6     0.8827221989631653   myRun 
3         Q0  34982904220237824  7     0.8695272207260132   myRun 
3         Q0  33711164877701120  8     0.8682869672775269   myRun 
3         Q0  34896269163896832  9     0.8675245046615601   myRun 
3         Q0  32809006015713280  10    0.867477297782898    myRun
```
##### Query 20

```
Topic_id  Q0  docno              rank  score                tag 
20        Q0  33356942797701120  1     0.9473576545715332   myRun 
20        Q0  32672996137111552  2     0.9401946067810059   myRun 
20        Q0  33983287403745281  3     0.9358925223350525   myRun 
20        Q0  34048315318345728  4     0.9331563711166382   myRun 
20        Q0  29958466130939904  5     0.9306454658508301   myRun 
20        Q0  29394885203206144  6     0.930182933807373    myRun 
20        Q0  34137228087136256  7     0.9294392466545105   myRun 
20        Q0  33290743200092160  8     0.9288673400878906   myRun 
20        Q0  29105489178529792  9     0.9250818490982056   myRun 
20        Q0  29341073989967872  10    0.9246830940246582   myRun 
```
#### Setting up & Execution

##### Download the BERT model
Run the following code  below in the terminal to download the BERT model:
- `pip3 install --user sentence_transformers`

##### Download the other necessary libraries
Run the following code below in the terminal to download the libraries:
- `pip3 install --user numpy`
- `pip3 install --user torch`

##### Run the program
- To run the BERT approach, call the `main.py` file inside the `/bert` folder after set up steps have been completed with `python3 main.py`


--------------------------------------------------------------------------------------------------------------------------------------------------------------


### Query Expansion (Approach 2 - Joshua Erivwo)

#### Functionality
In this section of the assignment, we were given the task to implement a query expansion based on pre-trained word embeddings and using other methods such as adding synonyms to the query if there is a similarity with more than one word in the query. We created functions in our `main.py` that would help expand the query. We also used `NLP` for word embedding that would later be implemented in calculating the similarity between two or more query words.
After expanding the query, the words would now contain synonyms used to retrieve and rank the tweet documents. For example, given the query "human," we can identify "human nature, human being" as a synonym for that particular query word and then add it to the query, matching both human and human nature or human beings. 

After the query is expanded, we follow the exact implementation we did earlier in `assignment 1` to calculate the Cosine Similarity between each Document & Query, respectively, and then store them in a dictionary in our results. After finding the `cosSim`, we then rank the tweets in the `dict()` from the highest to the lowest.


#### Discussion (Compared to Assignment 1)
- From Assignment 1: MAP: 0.1679    ;      P@10: 0.1833
- From query expansion : MAP: 0.1272    ;      P@10: 0.1149

The MAP and P@10 result from assignment one score is much better than the query expansion score. When expanding a query using synonyms, the recall is increased at the expense of precision.  Studies have been made that show how synonym query expansion can degrade a query performance rather than make it better. 
To have achieved a better result, the Rocchio algorithm's implementation would have helped improve the query expansion, even if it's a little bit. The Rocchio algorithm is mainly referred to as the relevancy feedback where it gets the top relevant document and then implements the query expansion using the top document.

In conclusion, the query expansions performed worse compared to the methods that were explored in this assignment. However, the benefit is that query expansion can be implemented with any of the approaches performed in this assignment. After the query's expansion, we can implement relevancy feedback that can calculate the cosine similarity again using the relevant documents

#### Algorithm and Data structure
Our implementation of the IR system for `Query expansion` was based on expanding the query by adding synonyms to the query if there was a similarity with more than one word in the query. We used `word embeddings` to find the similarity between the query word. The word embedding uses `natural language processing` to see the similarity for the query word. Afterward, we used `wordnet` to find the synonyms of the word terms that have similarities. 

The expansion of the query functions was implemented in the `main.py` file. Using the previously implemented IR  system in `Assignment 1`, we could retrieve and rank the newly created query and tweet documents.
Project Main Files

#### Project Main Files

##### `main.py`:
The `main.py` contains the primary and essential functions for executing the IR system. Some of the helper functions include the Preprocess file, spacy, and wordnet. The `main function` also contains a loaded `NLP` (Natural Language Processing), which creates the similarity for the query's words. We started by importing the `tweets` and the `queries` from the `assert` folder. Preprocessing is then performed on the import functions using the `filterSentence`. The `filterSentence` remains the same as the previous assignment, with a bit of modification made to it. The next step was to implement the functions that would expand the query using wordnet and NLP. The functions include:
-  `getSim(word, syn)`: Calculates the `similarity` between two words in the query. Word embedding is performed in this function to get the similarity for each query word.
- `getSyns(queryList)`: Finds the synonyms for each word in the query list.
- `canExpand(syn, queryList)`:  An helper function for expanding the query. It checks the similarity for each word in the query using the `getSim` function.
- `expand(query)`:  This is where we perform the query expansion. We first start by creating a`newQuery` and `syns` variables. The next step was to create a loop that checks for each synonym word in the list and then run the `canExpand` function for the synonym word and the query and ensure that the synonym word was not in the `newQuery` variable that was already created earlier. After appending the result into the newQuery, the variable is then returned.

Once the query's expansion is done, the `main()` function is then performed, which follows the same step in the previous assignment for the IR system using the newly expanded `query` and the `tweets`. To understand what is happening in the `main()` function, we created a set of print statements that would notify the user when the query is expanded and when the document's ranking is done. The user then gets informed of the creation of the result file.

##### `preprocess.py`:
The preprocessing file remains the same as the previous implementation, with a few modifications made in the `filterSentence` and the `importQuery`. 
In the `filterSentence`, we created two tokens, one with stemmed words and the other doesn't. The words that are not stemmed, are used in the query expansion for finding the words' synonyms. While in the `importQuery`, we also created two sets of query lists containing stemmed words and no stemmed words.


##### `result.py`:
This file contains the function for calculating the Cosimilarity values for the set of documents against each query and then ranks the similarity scores in descending order. Dictionary was used as our primary source for storing the query_index, retrieval, and query_length values. The function comprises mainly loops. At the start, we first calculated the occurrences of the token in each query. We then moved to calculate the TF-IDF and the length of the query. After getting the necessary calculations needed, we then moved to solving the CosSimalarity values and then ranking the document according to the specified order.


##### `write.py`:
This file contains a helper function that creates a table for each of the results generated in the `main.py` and then stores it in the `dist/query-expansion` directory.


#### Additional Libraries

##### Prettytable (`prettytable.py`):  
A helper library to format the output for the `Results.txt` file. Used in the implementation of the `write.py`.

#### Results for Query 3 & 20

##### Query 3
```
 1         Q0  31466391706017792  1     0.1998567335243553     myRun 
 1         Q0  30407444110778369  2     0.1477653234454287     myRun 
 1         Q0  34948668163362816  3     0.1386547082201938     myRun 
 1         Q0  34073394068590592  4     0.1290845236479003     myRun 
 1         Q0  32629073276571648  5     0.12373671801125216    myRun 
 1         Q0  32229379287289857  6     0.12161781019700534    myRun 
 1         Q0  30493951110676480  7     0.11940217949497257    myRun 
 1         Q0  30216589932503040  8     0.1191833587731748     myRun 
 1         Q0  29514474415198208  9     0.11699618729082718    myRun 
 1         Q0  30198105513140224  10    0.11629882192090524    myRun 
```
##### Query 20
```
 20        Q0  30649815905869824  1     0.3105204557845766     myRun 
 20        Q0  29803547608481792  2     0.23929276345833797    myRun 
 20        Q0  33356942797701120  3     0.23695303696737582    myRun 
 20        Q0  34082003779330048  4     0.1964380192155871     myRun 
 20        Q0  34066620821282816  5     0.1964380192155871     myRun 
 20        Q0  33752688764125184  6     0.1964380192155871     myRun 
 20        Q0  33695252271480832  7     0.1964380192155871     myRun 
 20        Q0  33580510970126337  8     0.1964380192155871     myRun 
 20        Q0  32866366780342272  9     0.1964380192155871     myRun 
 20        Q0  32269178773708800  10    0.1964380192155871     myRun 
```
#### Setting up & Execution

##### Download necessary libraries
Run the following code below in the terminal to download the libraries:
- `pip3 install --user spacy`
- `pip3 install --user wordnet`
- `python3 -m spacy download en_core_web_lg`

##### Run the program
- To run the QueryExpansion approach, call the `main.py` file inside the `/query-expansion` folder after set up steps have been completed with `python3 main.py`

--------------------------------------------------------------------------------------------------------------------------------------------------------------

###  (Approach 3 - Dmitry Kutin) 

See Facebook Research's [InferSent project](https://github.com/facebookresearch/InferSent), an implementation of a pre-trained English sentence encoder from [their paper](https://arxiv.org/abs/1705.02364) and SentEval evaluation toolkit.

#### Functionality

In the InferSent approach, we first have to download the provided pre-trained model of 300 dimension vectors for the first 50, 000 most common words, and InferSent's sentence encoder.

After this, similar to our implementation in Assignment 1, we import & format the list of test queries and documents, tokenized & rebuilt into sentence format for input to the InferSent library. We opted to use an improved tokenization implementation from our first assignment, rather than using InferSent's built-in tokenizer and stemmer that uses NLTK.

We then use the InferSent model to embed the queries and documents within each sentence. This step is done by converting our dictionaries of documents and queries into lists and encoding them with InferSent's pre-trained model.

Now that our model is set up, we simply loop through all of our imported queries and documents, and calculate the Cosine Similarity between each Document & Query respectively, and store them to our dictionary of results of the format:

` query_id : {document_id : CosSim, ...} `

for ease of iteration to rank the top 1000 documents. 

We then use the built-in python function `sorted` to sort all 50, 000 documents for each query in descending order, and grab the first 1000 indices of the resulting list, and save them to our results file (`dist/infersent/infersent_results.txt`) for further evaluation using `trec_eval`. 

#### Discussion (Compared to Assignment 1)

The scores for `MAP` and `P@10` for Assignment 1 we achieved were `0.1679` & `0.1833` respectively, compared to `0.1970` & `0.2735` for the InferSent implementation -- A significant improvement.

InferSent uses a pre-trained model of 300 feature dimensions, allowing for better classification of each document for the query, which we suspect is the reason for our improved scores. Using the `numpy` library, we're able to take the Cosine Similarity more effectively using each query and document's feature dimensions. We were able to see an increase in performance when using GPU processing (Taking ~2.3 minutes to rank & retrieve all documents), even having to calculate the dot product between 2 vectors of dimension 300, over 2.5 million times (50 queries * 50k documents).

We saw an increase in similarity scores between the query and document vectors, seeing some high similarity scores (`0.82` avg. score top result), and the lowest being no lower than `0.5`. Compared to our Assignment 1 implementation, we had limited consistency in similarity scores, and fewer feature dimensions using just word similarity using the sent2vec implementation. These improvements on similarity scores can also be attributed to our improvement in `P@10` and `MAP`.

#### Algorithms and Datastructures

Our implementation of the IR system for InferSent utilizes dictionaries and NumPy arrays for the ranking and retrieval of the query and document embeddings. As discussed in the previous section, we make use of dictionaries when using our implementation for preprocessing and retrieval to take advantage of the fast indexing the `dict` data structure has to offer. 

When utilizing the InferSent library, we then had to convert our dictionaries to NumPy arrays to remain consistent with the InferSent libraries, though this conversion is made easy with python built-in functions.

#### Project Specific Files

##### `main.py`

Responsible for handling the main execution of the program. the main file loads the sentence encoder & pre-trained model of the most common 50, 0000 sentences used for InferSent, with 300 feature dimensions, and initializes the InferSent model for query and document embedding.

Using an improved implementation from our first assignment, we also import all the queries and documents subject to preprocessing and convert them to lists, to be encoded using our InferSent model.

After the initialization process, we then loop through all of our encoded queries and documents, and calculate the Cosine Similarity using `numpy` functions for vector dot products over the previously mentioned 300 feature dimensions, and save this result to our `results` dictionary.

After the Cosine Similarity has been calculated for all documents for a single query, we sort the documents in descending order and capture the first 1000 indices as per the assignment guidelines. 

When each document for each query has been properly ranked, we then write our results using a helper function in `write.py` to achieve our final `Results.txt` for further evaluation with `trec_eval`.

##### `preprocess.py`

A collection of helper functions used in `main.py` for retrieval, formatting, and tokenization in preparation to encode our queries and documents using InferSent.

importTweets(): imports the tweets from the collection. We first started by opening the text files, then we filter the file using our filterSentence function.

importQuery(): imports query from the collection. Utilizes the same process as the importTweet().

filterSentence(sentence): Filters sentences from tweets and queries. This function builds a list of stopwords and then tokenizes each word in the sentences by removing any numerics, links, single characters, punctuation, extra spaces, or stopwords contained in the list. Each imported tweet and query runs through the NLTK's stopword list, our custom stopword list that included the URLs and abbreviations, and the provided stopword list. After this step, each word is tokenized and stemmed with Porter stemmer. Under the additional libraries section, we discussed in-depth the use of tokenization, stopwords, and porter stemmer. If this function is in bertMode a string is returned with all the remaining words otherwise a tokenized list of all the remaining words is returned. We then rebuild the tokenized string for encoding using InferSent.

##### `write.py`

The `writeResults` is a helper function that creates a table for each of the results generated in the `main.py` and storing the text file in `dist/infersent` directory.

#### Additional Libraries

##### Prettytable (prettytable.py):

A helper library to format the output for the Results.txt file. Used in the implementation of the write.py.

##### InferSent (`models.py`)

The model for InferSent resides here, pulled from [Facebook Research's Github]( https://github.com/facebookresearch/InferSent).

#### Results for Query 3 & 20

##### Query 3

```
 3         Q0  32273316047757312  1     0.79771817  myRun 
 3         Q0  29296574815272960  2     0.7946324   myRun 
 3         Q0  32383831071793152  3     0.7838675   myRun 
 3         Q0  29278582916251649  4     0.7818041   myRun 
 3         Q0  29323772183969793  5     0.7660473   myRun 
 3         Q0  33711164877701120  6     0.76154566  myRun 
 3         Q0  29046646381744129  7     0.75826454  myRun 
 3         Q0  32469924240695297  8     0.75596863  myRun 
 3         Q0  32488312107175936  9     0.75485104  myRun 
 3         Q0  32878302150529024  10    0.7486612   myRun 
 ```

##### Query 20

```
 20        Q0  33356942797701120  1     0.840229    myRun 
 20        Q0  34056572233580544  2     0.833139    myRun 
 20        Q0  34048315318345728  3     0.833139    myRun 
 20        Q0  29913837511647232  4     0.8098888   myRun 
 20        Q0  33960436810387456  5     0.80659086  myRun 
 20        Q0  29022284970729473  6     0.79709256  myRun 
 20        Q0  29902271479287808  7     0.79665387  myRun 
 20        Q0  30692965663899648  8     0.7948294   myRun 
 20        Q0  30075676803465217  9     0.7934148   myRun 
 20        Q0  30283063699177472  10    0.791473    myRun 
```

#### Setting up & Execution

##### Download necessary libraries
Run the following code below in the terminal to download the libraries:
- `pip3 install --user numpy`
- `pip3 install --user torch`
- `pip3 install --user nltk`

After installing all required packages, we will also need to download and format the sentence encoder & pre-trained model. 

**From the `infersent` directory:**

Download GloVe (V1):
```
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

Download the Sentence Encoder:
```
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

##### Run the program
- To run the Infersent approach, call the `main.py` file inside the `/infersent` directory after set up steps have been completed with `python3 main.py`.

## Final Result Discussion (InferSent)

Out of the Advanced IR methods we'd experimented with, we found that the InferSent (sent2vec) implementation achieved the best results when being evaluated by the `trec_eval` script. 

The `Results.txt` file can be found at the root of the directory. 

For the InferSent approach, all evaluation files for `trec_eval`, and `trec_eval_all_queries`, (as well as a backup of `Results.txt`) can be found in the `dist/infersent` directory. 

We were able to achieve scores for:
- MAP: `0.1970`
- P@10: `0.2735`

a considerable improvement from our Assignment 1. 

For more information regarding the InferSent approach, see above for the full breakdown in the `InferSent (Approach 3)` section.
