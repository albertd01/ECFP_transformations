from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
import torch
from utils.feature_extraction import mol_to_duvenaud_graph
import copy
import numpy as np


class DuvenaudDataset(InMemoryDataset):
    def __init__(self, name, transform=None, pre_transform=None):
        self.name = name.lower()
        self.supported = ['esol', 'bace', 'lipo']
        if self.name not in self.supported:
            raise ValueError(f"Dataset '{self.name}' not supported. Choose from {self.supported}")
        
        self.smiles_list = []
        self.labels = []

        super().__init__(root=f"data/{self.name}", transform=transform, pre_transform=pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f'{self.name}_duvenaud.pt']

    def process(self):
        dataset = MoleculeNet(root=self.root, name=self.name.upper() if self.name == 'esol' else self.name.capitalize())
        data_list = []

        for data in dataset:
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_duvenaud_graph(mol)
            graph.y = data.y
            graph.smiles = smiles

            self.smiles_list.append(smiles)
            self.labels.append(data.y.item())
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def corrupt_labels(self, task_type: str, seed: int = 42):
        # Set reproducibility
        np.random.seed(seed)

        # Clone the dataset
        corrupted_data_list = []
        ys = [self.get(i).y.item() for i in range(len(self))]

        if task_type == "regression":
            y_min, y_max = min(ys), max(ys)
            print(f"[Corruption] Label range: {y_min:.3f} to {y_max:.3f}")

        for i in range(len(self)):
            data = self.get(i).clone()
            if task_type == "regression":
                data.y = torch.tensor([np.random.uniform(y_min, y_max)], dtype=torch.float)
            else:
                data.y = torch.tensor([np.random.randint(0, 2)], dtype=torch.float)
            corrupted_data_list.append(data)
            
        if task_type == "classification":
            labels = [self.get(i).y.item() for i in range(len(self))]
            counts = {0: labels.count(0.0), 1: labels.count(1.0)}
            print(f"[Corruption] Original label distribution: {counts}")

        # Replace internal data
        self.data, self.slices = self.collate(corrupted_data_list)
        print("[Corruption] Dataset labels have been randomized.")
        
        if task_type == "classification":
            labels = [self.get(i).y.item() for i in range(len(self))]
            counts = {0: labels.count(0.0), 1: labels.count(1.0)}
            print(f"[Corruption] Random label distribution: {counts}")
        
