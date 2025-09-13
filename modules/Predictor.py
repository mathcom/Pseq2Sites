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