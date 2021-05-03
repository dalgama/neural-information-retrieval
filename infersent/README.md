# Setup Process for InferSent

See: https://github.com/facebookresearch/InferSent

## Dependencies

This code is written in python. Dependencies include:

* Python 2/3
* [Pytorch](http://pytorch.org/) (recent version)
* NLTK >= 3

## Download word vectors

Download [GloVe](https://nlp.stanford.edu/projects/glove/) (V1):
```bash
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

## Download sentence encoder

### InferSent models trained with GloVe[147MB]:
```bash
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

## Download necessary libraries
Run the following code below in the terminal to download the libraries:
- `pip3 install --user numpy`
- `pip3 install --user torch`
- `pip3 install --user nltk`

After installing all required packages, we will also need to download and format the sentence encoder & pre-trained model. 

## Run the program
- To run the Infersent approach, call the `main.py` file inside the `/infersent` directory after set up steps have been completed with `python3 main.py`.