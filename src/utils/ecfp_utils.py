from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import hashlib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def hash_atom_info(atom):
    features = (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
    )
    s = str(features)
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h, 16) & 0xFFFFFFFF

def hash_tuple(t):
    h = hashlib.sha1(str(t).encode('utf-8')).hexdigest()
    return int(h, 16) & 0xFFFFFFFF


# From scratch implemenetation
def algorithm1_duvenaud(mol, radius=2, nBits=2048):
    fp_bits = set()
    atom_ids = {}
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_ids[idx] = hash_atom_info(atom)

    for iteration in range(radius):
        new_ids = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            info = (atom_ids[idx],) + tuple(
                atom_ids[neighbor.GetIdx()]
                for neighbor in atom.GetNeighbors()
            )
            env_id = hash_tuple(info)
            fp_bits.add(env_id % nBits)
            new_ids[idx] = env_id
        atom_ids = new_ids

    fp_array = np.zeros(nBits, dtype=int)
    for bit in fp_bits:
        fp_array[bit] = 1
    return fp_array

def get_custom_invariants(mol):
    invariants = []
    for atom in mol.GetAtoms():
        invariants.append(hash_atom_info(atom))
    return invariants

# RDKit implementation
def compute_ecfp_array(smiles: str, radius: int, nBits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    invariants = get_custom_invariants(mol)  
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol, customAtomInvariants=invariants)

    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def compute_ecfp_fp(smiles: str, radius: int, nBits: int, count_fp = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    invariants = get_custom_invariants(mol) #doesnt seem to make a difference with this feature subset
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return mfpgen.GetFingerprint(mol, customAtomInvariants=invariants)




def compute_ecfp_bit_vectors(smiles_list, radius=2, nBits=2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits
    )

    fp_array = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")
        invariants = get_custom_invariants(mol)
        fp = mfpgen.GetFingerprint(mol, customAtomInvariants=invariants)
        arr = np.zeros((nBits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array.append(arr)

    return np.array(fp_array)

def compute_ecfp_count_vectors(smiles_list, radius=2, nBits=2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits
    )

    fp_array = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")
        invariants = get_custom_invariants(mol)
        fp = mfpgen.GetCountFingerprint(mol, customAtomInvariants=invariants)
        arr = np.zeros((nBits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array.append(arr)

    return np.array(fp_array)

def compute_algorithm1_fps(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = algorithm1_duvenaud(mol, radius=radius, nBits=nBits)
        fps.append(fp)
    return np.array(fps)


    


