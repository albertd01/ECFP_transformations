from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import hashlib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

'''Although Algorithm 1 in Duvenaud et al. is described as a simplified analog of ECFP, 
we find that its practical output differs significantly from RDKit’s canonical ECFP. 
Using aligned atom features and hashing, 
our implementation of Algorithm 1 yielded near-zero Tanimoto similarity to RDKit ECFP across the ESOL dataset. 
This suggests that claims about similarity in the NGF paper may rely on using RDKit's implementation, 
not Algorithm 1 as described.'''

def hash_atom_info(atom):
    features = (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
        #atom.IsInRing()
    )
    s = str(features)
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h, 16) & 0xFFFFFFFF  # 32-bit

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

def compute_algorithm1_fps(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = algorithm1_duvenaud(mol, radius=radius, nBits=nBits)
        fps.append(fp)
    return np.array(fps)

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
    

def compute_tanimoto(fp1, fp2):
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    return intersection / union


'''from rdkit.Chem import MolFromSmiles

smiles = 'CCO' 
mol = MolFromSmiles(smiles)
fingerprint_from_scratch = algorithm1_duvenaud(mol, radius=2, nBits=1024)
fingerprint_rdkit = compute_ecfp_array(smiles, radius=2, nBits=1024)
fingerprint_paper_repo = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
arr = np.zeros((1024,), dtype=int)
DataStructs.ConvertToNumpyArray(fingerprint_paper_repo, arr)

print("From scratch bits:", np.where(fingerprint_from_scratch == 1)[0])
print("RDKit bits:", np.where(fingerprint_rdkit == 1)[0])
print("Paper repo bits:", np.where(arr == 1)[0])

print(f"Tanimoto similarity between implementation of Algorithm1 and rdkit (same atom features): {compute_tanimoto(fingerprint_from_scratch, fingerprint_rdkit):.4f}")
print(f"Tanimoto similarity between implementation of Algorithm1 and paper github: {compute_tanimoto(fingerprint_from_scratch, arr):.4f}")
print(f"Tanimoto similarity between implementation of rdkit and paper github: {compute_tanimoto(fingerprint_rdkit, arr):.4f}")'''