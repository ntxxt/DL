import torch
import os
from collections import OrderedDict
import numpy as np
from rdkit import Chem

import networkx as nx

#compute atom features
def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

#atom type, degree, H
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


#generating molecular graph from smile
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    ligand_size = mol.GetNumAtoms()
    
    #node feature
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    
    #edges(bond)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()

    
    #generate adjacency matrix
    edge_index = []
    mol_adj = np.zeros((ligand_size, ligand_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    index_row, index_col = np.where(mol_adj >= 1)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # ? why return three values
    return ligand_size, features, edge_index

smile = 'CC(=O)OC1=CC=CC=C1C(=O)O '
smile_graph = smile_to_graph(smile)
print(smile_graph)