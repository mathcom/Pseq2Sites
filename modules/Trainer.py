import os
import tqdm
import torch
import numpy as np

from .Encoders import Pseq2Sites
from .helpers import prepare_prots_input


class Pseq2SitesTrainer:
    def __init__(self, config):
        self.config = config
        
        # build model
        self.model = Pseq2Sites(self.config)
        self.modle = self.model.cuda()
        
    def train(self, train_loader, validation_loader, save_path):
        # define optimizer
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr = 1e-3,
            weight_decay = 0.01
        )
        
        ## init
        history = []
        best_eval_loss = np.inf
        total_epochs = self.config["train"]["epochs"]
        
        ## run
        for epoch in tqdm.trange(total_epochs):
            self.model.train()
            train_losses = 0.
            for batch in train_loader:
                # inputs
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = prepare_prots_input(self.config, batch)
                                
                # forward
                pred_BS, _, _ = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)
                
                # loss
                loss = self.get_multi_label_loss(pred_BS, binding_sites)
                train_losses += loss.item()
                
                # backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            ## Evaluation on Validation set
            val_losses = self.evaluate(validation_loader)    
            
            ## Average
            train_losses_avg = train_losses / len(train_loader)
            val_losses_avg = val_losses / len(validation_loader)
            
            ## Logging
            history.append({
                "Epoch": epoch,
                "Loss_tr": train_losses_avg,
                "Loss_va": val_losses_avg
            })
            print(", ".join([
                f"Epoch: {epoch} / {total_epochs}",
                f"Train Loss: {train_losses_avg:.3f}",
                f"Validation Loss: {val_losses_avg:.3f}",
            ]))
            
            ## Save
            if best_eval_loss > val_losses_avg:
                ## update score
                best_eval_loss = val_losses_avg
                
                ## update ckpt
                self.save_checkpoint(
                    save_path,
                    "Pseq2Sites.pth",
                    state_dict = self.model.state_dict(),
                    optimizer = self.optim.state_dict()
                )
                
        return history

    
    def evaluate(self, loader):
        losses = 0.
        
        with torch.no_grad():
            self.model.eval()
            
            for batch in loader:
                # prepare input
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = prepare_prots_input(self.config, batch)
                
                # forward
                pred_BS, _, _ = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)
                
                # cal loss
                loss = self.get_multi_label_loss(pred_BS, binding_sites)
                losses += loss.item()

            return losses                

        
    def get_multi_label_loss(self, predictions, labels):
        weight = self.calculate_weights(labels)

        loss_ft = torch.nn.BCEWithLogitsLoss(weight = weight)
        loss = loss_ft(predictions, labels)        

        return loss        

    
    def calculate_weights(self, labels):
        labels_inverse = torch.abs(labels - torch.ones(labels.size()).cuda())
        
        negative_labels = labels_inverse
        
        P = torch.sum(labels)
        N = torch.sum(negative_labels)

        P_weights = (P + N + 1) / (P + 1)
        N_weights = (P + N + 1) / (N + 1)

        weights = torch.multiply(labels, P_weights) + torch.multiply(negative_labels, N_weights)
        
        return weights 

    
    def save_checkpoint(self, dir, name, **kwargs):
        state = {}
        state.update(kwargs)
        filepath = os.path.join(dir, name)
        torch.save(state, filepath)          