from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import time

# break protein into 3AAs 
def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]

# define an iteratable for training seq2vec models
# inputs: collections of AAs, k
class Corpus(object):
    def __init__(self, sequences, ngram):
        self.sequences = sequences
        self.ngram = ngram

    def __iter__(self):
        for sequence in self.sequences:
            yield seq_to_kmers(sequence, self.ngram)

# get target protein embeddings with pre-trained model (num_word,100)
def get_protein_embedding(model,protein):
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

#process all human protein
FASTA = 'uniprot-compressed_true_download_true_format_fasta_query__28proteome-2022.10.04-12.25.23.98.fasta'
in_file = open(FASTA, 'r')
data=''
name_list=[]
seq_list=[]
for line in in_file:
    line = line.strip()
    for i in line:
        if i== '>':
            name_list.append(line)
            if data:
                seq_list.append(data)
                data=''
            break
        else:
            line=line.upper()
    if all([k==k.upper() for k in line]):
            data+= line

#train the model
start = time.time()
sent_corpus = Corpus(seq_list,3)
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=6)
model.build_vocab(sent_corpus)
model.train(sent_corpus,epochs=10,total_examples=model.corpus_count)
end = time.time()
print(end-start)
model.save("human.model")


"""
model = Word2Vec.load("human.model")
vector = get_protein_embedding(model,seq_to_kmers("MSPLNQSAEGLPQSDC", k=3))
print(vector.shape)
"""




