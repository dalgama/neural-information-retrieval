import os.path
from os import path
from prettytable import PrettyTable

# Ordered Ranking[query_no] = {d1 : cosSim, d2 : cosSim, ...}
def resultFileCreation(Rankings, Bert = False):
    # File path of the saved file
    file_path = "../dist/bert/bert_results.txt"

    # Rename the file when working with the Bert results
    if (Bert == True):
        file_path = "../dist/bert/bert_results.txt"

    # Check if the results file exists
    if(path.exists(file_path) == False):
        # Initialize the object passing the table headers
        rTable = PrettyTable(['Topic_id','Q0', 'docno','rank','score','tag'])
        # Align the table to the left of the txt file.
        rTable.align='l'
        # Remove borders of the table
        rTable.border=False

        # Add rows with the data to the table.
        for query_num,value in Rankings.items():
            # List used to store all the element of row which will be added to the table once populated.
            list = []
            # Re-initialize the ranking for each query.
            ranking = 1
            for doc_num, cosSim in value.items():
                # Column order ['Topic_id/queryno','Q0', 'docno','rank','score','tag']
                list = [query_num,"Q0",doc_num,ranking,cosSim,"myRun"]
                # increment the ranking
                ranking +=1
                # Adding the row of data to the table
                rTable.add_row(list)

        # Making all the data in the table into a string.
        table_text = rTable.get_string()

        # Write table to the file after populating the table.
        f = open(file_path,"w+")
        f.write(table_text)

        # Close Results file.
        f.close()
        return
    # Remove file if Results file exists and recall the function.
    else:
        os.remove(file_path)
        resultFileCreation(Rankings,Bert)


