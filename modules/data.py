"""
Pseq2Sites
Copyright (C) 2025 Jonghwan Choi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from torch.utils.data import Dataset, DataLoader

def Dataloader(dataset, batch_size, shuffle = True, drop_last = False): 
    
    data_loader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle = shuffle,
                    drop_last = drop_last,
                    collate_fn = lambda x:x,
                    pin_memory = True
                )
                
    return data_loader
    
class PocketDataset:
    def __init__(self, IDs, feats, seqs, binding_sites = None):
        self.IDs = IDs
        self.feats = feats
        self.seqs = seqs
        self.binding_sites = binding_sites
 
    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):
        if self.binding_sites is not None:
            return self.IDs[idx], self.feats[idx], self.seqs[idx], self.binding_sites[idx]
        else:
            return self.IDs[idx], self.feats[idx], self.seqs[idx] 
