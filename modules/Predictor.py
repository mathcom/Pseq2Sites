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

import torch
from .Encoders import Pseq2Sites
from .helpers import prepare_prots_input


class Pseq2SitesPredictor:
    def __init__(self, config):
        self.config = config
        
        # build model
        self.model = Pseq2Sites(self.config)
        self.modle = self.model.cuda()
        
        
    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath)["state_dict"])
        return self
        
        
    def predict(self, loader):
        predictions = list()
        
        with torch.no_grad():
            self.model.eval()
            
            for batch in loader:
                # prepare input
                aa_feats, prot_feats, prot_masks, position_ids, chain_idx = prepare_prots_input(self.config, batch, training = False)
                
                # forward
                pred_BS, _, _ = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)              
                pred_BS = pred_BS * prot_masks
                predictions.append(pred_BS.detach())
            
        predictions = torch.cat(predictions)
            
        return torch.sigmoid(predictions).cpu().numpy()