from cProfile import label
from ctypes.wintypes import tagRECT
from multiprocessing.context import ForkProcess
import os
from re import X
import numpy as np
from numpy import save
from tkinter import N 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data 
from torch.utils.data import DataLoader
""" 
from rdkit import Chem
import gensim
import pickle
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec



p1. generate ligand vector representation
#import torch
#import torch.nn as nn
GENSIM4 = gensim.__version__ > "4"  
#print(GENSIM4)

file_path = '/Users/xiaotongxu/Downloads/human_data.csv'

file = pd.read_csv(file_path,header=0)
#print('Protein' + str(file.iloc[43,3]) + '.pt')
fastas = file['fasta'][:500]
labels = file['label'][:500]
#print(len(fastas))
#print(file.head())

s = ''
for i in range(len(fastas)):
    lineone = f"\n>Protein{i+1}\n"
    s  += lineone
    s += fastas[i]
print(s)

with open('fasta.test', 'w') as f:
    f.write(s)

#smile -> mol -> substracture_sentence -> vector (500, 300) (mol, embedding)
smiles = file['smile'][:500]
mols = [Chem.MolFromSmiles(i) for i in smiles]
model = Word2Vec.load('/Users/xiaotongxu/tests/mol2vec/examples/models/model_300dim.pkl')
Molecular_sentence = [mol2alt_sentence(mol, radius=1) for mol in mols]
mol2vec = [DfVec(x) for x in sentences2vec(Molecular_sentence, model, unseen='UNK')]
X = np.array([x.vec for x in mol2vec])
save('ligand_embed.npy', X)
#ligand_embed = torch.from_numpy(X)
#print(ligand_embed.size())
#print(X[342])
"""


"""
#p2.search for max_len of input protein embeddings (longest  within proteins 1023) torch.size([a, 1280])
# use padding to make the protein vector all same shape 
files = os.listdir('/Users/xiaotongxu/Downloads/test500_protein')
#print(files[40])

len_  = []
for i in files:
    protein_embed = torch.load(os.path.join('/Users/xiaotongxu/Downloads/test500_protein/', i))['representations'][33]
    print(protein_embed.size())
    len = protein_embed.size()[0]
    len_.append(len)
print(max(len_))    

example_protein = torch.load('/Users/xiaotongxu/Downloads/test500_protein/Protein23.pt')
embed = example_protein['representations'][33]
target_len = 1023
embed = F.pad(embed, (0,0,int(target_len-embed.size()[0]),0))
m = nn.Conv1d(1023, 33, 3, stride=2)
print(embed.size())
output = m(embed)
   
"""
#p3. protein dataset class
class ProteinDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        protein_path = os.path.join(self.root_dir, 'Protein' + str(self.annotations.iloc[index, 3]) + '.pt')
        protein_embed = torch.load(protein_path)['representations'][33]
        target_len = 1023
        protein_embed = F.pad(protein_embed, (0,0,int(target_len-protein_embed.size()[0]),0))
        #y_label = torch.tensor(self.annotations.iloc[index, 2])
        if self.transform:
            protein_embed = self.transform
        return protein_embed

protein_dataset = ProteinDataset(csv_file='/Users/xiaotongxu/Downloads/human_data.csv', root_dir='/Users/xiaotongxu/Downloads/test500_protein',)
protein_dataloader = DataLoader(protein_dataset, batch_size=4,
                        shuffle=False, num_workers=0)

#p4 ligand dataset
class LigandDataset(Dataset):
    def __init__(self, npy_file, transforms=None):
        self.data = torch.from_numpy(np.load(npy_file))
        self.transforms = transforms
    
    def __len__(self):
        return self.data.size()[0]  #500 embeded smile vectors
    
    def __getitem__(self, index):
        smile_embed = self.data[index]
        if self.transforms:
            smile_embed = self.transforms(smile_embed)
        return smile_embed
ligand_dataset = LigandDataset(npy_file='/Users/xiaotongxu/Downloads/ligand_embed.npy',transforms=None)
ligand_dataloader = DataLoader(ligand_dataset, batch_size=4, shuffle=False, num_workers=0)

#P5 combine two dataset object into a tuple
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

train_loader = DataLoader(
             ConcatDataset(
                 ligand_dataset,
                 protein_dataset
             ),
             batch_size=4, shuffle=False)

a = ConcatDataset(
                 ligand_dataset,
                 protein_dataset
             )

#pass protein
