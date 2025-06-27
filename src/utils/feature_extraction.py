import torch
from torch_geometric.data import Data
from rdkit import Chem

# ADAPTED FROM: https://github.com/HIPS/neural-fingerprint/blob/master/neuralfingerprint/features.py
# LINES 7 -55
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
DEGREES = list(range(5))  
NUM_HS = list(range(5))   
VALENCES = list(range(5)) 

BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

def one_hot(value, choices):
    vec = [0] * len(choices)
    idx = choices.index(value) if value in choices else -1
    if idx >= 0:
        vec[idx] = 1
    return vec

def mol_to_duvenaud_graph(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        features = one_hot(symbol, ATOM_SYMBOLS) + \
                   one_hot(atom.GetDegree(), DEGREES) + \
                   one_hot(atom.GetTotalNumHs(), NUM_HS) + \
                   one_hot(atom.GetImplicitValence(), VALENCES) + \
                   [int(atom.GetIsAromatic())]
        atom_features.append(features)
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = one_hot(bond.GetBondType(), BOND_TYPES) + \
                    [int(bond.GetIsConjugated()), int(bond.IsInRing())]

        edge_index += [[i, j], [j, i]]
        edge_features += [bond_feat, bond_feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
