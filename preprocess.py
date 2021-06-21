from collections import defaultdict
import os
import pickle
import numpy as np
from rdkit import Chem
import pandas as pd

#主要参考论文"Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)" 提供的代码

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)



def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


radius = 2
train_seq_label = pd.read_csv('Data/train.csv')
train_sequence = train_seq_label['sequence'].values
train_label = train_seq_label['label']

test_seq_label = pd.read_csv('Data/test.csv')
test_sequence = test_seq_label['sequence']
test_label = test_seq_label['label']

sequences = train_sequence.append(test_sequence)
sequence_list = sequences.values

labels = train_label.append(test_label)
label_list = labels.values

N = len(sequence_list)

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

compounds, adjacencies, labels = [], [], []

for i in range(len(sequence_list)):
    #         print('/'.join(map(str, [no+1, N]))) #str() 将数据变为字符串; 'sep'.join(seq) 用特定符号连接字符或字符列表
    #         emumerate 返回数据下标和数据

    #         smiles, sequence, interaction = data.strip().split()
    sequence = sequence_list[i]
    label = label_list[i]

    mol = Chem.rdmolfiles.MolFromSequence(sequence)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    compounds.append(fingerprints)

    adjacency = create_adjacency(mol)
    adjacencies.append(adjacency)

    labels.append(np.array([float(label)]))

dir_input = ('Data/rdkit/radius2/')
os.makedirs(dir_input, exist_ok=True)

np.save(dir_input + 'compounds', compounds)
np.save(dir_input + 'adjacencies', adjacencies)
np.save(dir_input + 'labels', labels)
dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

print('The preprocess of dataset has finished!')