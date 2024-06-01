from typing import Any
from datasets import load_from_disk

class DataLoader():

    def __init__(self, dataset_name, seed=None) -> None:
        self.dataset_name = dataset_name
        if dataset_name == "math":
            dataset = load_from_disk("data_sets/math_dataset_algLin")
        if seed != None:
            dataset = dataset.shuffle(seed=seed)
        self.dataset = dataset
        
    
    def __getitem__(self, __name: str) -> Any:
        if self.dataset_name == "math":
            return {'question':self.dataset[__name]['question'].split("b'")[1].split(r"\n")[0],
                    'answer':self.dataset[__name]['answer'].split("b'")[1].split(r"\n")[0]}
        
    def __len__(self) -> int:
        return len(self.dataset)