#!/usr/bin/env bash
FULL_EVAL=""
QRELS=""
RESULTS=""
DESTINATION=""

print_usage() {
    echo "Options:"
    echo "-q <file name>           specify the QRels file."
    echo "-r <file name>           specify the Results file."
    echo "-d <file name>           specify a directory to store output in."
    echo "-f [optional]            run full query evaluation."
    exit 0
}

while getopts 'fq:r:d:' flag; do
    case "${flag}" in
        q) QRELS="${OPTARG}" ;;
        r) RESULTS="${OPTARG}" ;;
        d) DESTINATION="${OPTARG}" ;;
        f) FULL_EVAL="-q" ;;
        *) print_usage
        exit 1 ;;
    esac
done

if [[ ! -e "$RESULTS" ]]
then
    echo "$RESULTS does not exist."
    exit 1
fi

if [[ ! -e "$QRELS" ]]
then 
    echo "$QRELS does not exist."
    exit 1
fi

if [[ -e "$DESTINATION" ]]
then
    echo "$DESTINATION exists. Please remove/backup before running"
    exit 1
fi

## Run the evaluation script.
./lib/trec_eval $FULL_EVAL $QRELS $RESULTS >> $DESTINATION
