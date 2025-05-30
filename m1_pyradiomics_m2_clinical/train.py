import torch
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import numpy as np

import time
import os 

import matplotlib.pyplot as plt
import sklearn.metrics 

class Fusion_Solver():
    def __init__(self, rep, i, fusion_model, loss, outdim_size, epoch_num, batch_size, learning_rate, reg_par, early_stopping_epoch, device=torch.device('cpu')):
        self.rep = rep
        self.i = i
        self.device = device
        
        self.model = fusion_model 
        self.model.to(self.device)

        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

        self.outdim_size = outdim_size
        self.early_stopping_epoch = early_stopping_epoch

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, figure_dir = None, checkpoint='fusion_checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = np.Inf
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses_epoch = [] 
        val_losses_epoch = []
        prior_to_train_loss, prior_to_train_outputs = self.test(tx1, tx2)
        output1 = prior_to_train_outputs[0]
        output2 = prior_to_train_outputs[1]
        num_columns = output1.shape[1]
        
        early_stop_count = 0

        for epoch in range(self.epoch_num):
            if early_stop_count < self.early_stopping_epoch:
                train_losses = []
                print(f"Fusion epoch - {epoch}/{self.epoch_num}")
                self.model.train()
                batch_idxs = list(BatchSampler(RandomSampler(
                    range(data_size)), batch_size=self.batch_size, drop_last=False))
                batch_idxs = [batch for batch in batch_idxs if len(batch) > 1]
                for batch_idx in batch_idxs:
                    self.optimizer.zero_grad()
                    batch_x1 = x1[batch_idx, :]
                    batch_x2 = x2[batch_idx, :]
                    o1, o2 = self.model(batch_x1, batch_x2)

                    loss = self.loss(o1, o2)
                    train_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                train_loss = np.mean(train_losses)
                train_losses_epoch.append(train_loss) 

                if vx1 is not None and vx2 is not None:
                    with torch.no_grad():
                        self.model.eval()
                        val_loss, val_outputs = self.test(vx1, vx2)
                        val_losses_epoch.append(val_loss)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop_count = 0
                            torch.save(self.model.state_dict(), checkpoint)
                        else:
                            early_stop_count += 1
                    
                    self.scheduler.step(val_loss)
                else:
                    torch.save(self.model.state_dict(), checkpoint)
            
        figure_dir = os.path.join(figure_dir, 'fusion_part')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        plt.plot(train_losses_epoch, label='Training loss')
        plt.plot(val_losses_epoch, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(figure_dir, 'split' + str(self.rep) + '_fold' + str(self.i) + 'losses.png'))
        plt.clf()

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss, output = self.test(vx1, vx2)

        if tx1 is not None and tx2 is not None:
            loss, output = self.test(tx1, tx2)

        return self.model

    def test(self, x1, x2):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            return np.mean(losses), outputs

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


class Downstream_Solver():
    def __init__(self, rep, i, model, losses, epoch_num, batch_size, learning_rate, reg_par, early_stopping_epoch, device=torch.device('cpu')):
        self.rep = rep
        self.i = i
        self.device = device
        self.model = model
        self.model.to(self.device)
        
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.losses = losses 
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = reg_par)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

        self.early_stopping_epoch = early_stopping_epoch

    def finetune(self, 
            x1, x2, label, 
            vx1=None, vx2=None, vlabel=None, 
            tx1=None, tx2=None, tlabel=None, 
            figure_dir = None,
            checkpoint='checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated w.r.b. to outcome 
        dim=[batch_size, feats]
        """

        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = np.inf
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses_epoch = [] 
        val_losses_epoch = []
        early_stop_count = 0
        for epoch in range(self.epoch_num):
            if early_stop_count < self.early_stopping_epoch:
                train_losses = [] ##
                epoch_start_time = time.time()
                self.model.train()
                batch_idxs = list(BatchSampler(RandomSampler(
                    range(data_size)), batch_size=self.batch_size, drop_last=False))
                batch_idxs = [batch for batch in batch_idxs if len(batch) > 1]
                for batch_idx in batch_idxs:
                    self.optimizer.zero_grad()
                    batch_x1 = x1[batch_idx, :]
                    batch_x2 = x2[batch_idx, :]

                    batch_label = label[batch_idx, :]
                    _, _, pred_logits = self.model(batch_x1, batch_x2)
                    loss = self.losses['loss'](pred_logits, batch_label)
                    train_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                train_loss = np.mean(train_losses)
                train_losses_epoch.append(train_loss) 

                if vx1 is not None and vx2 is not None and vlabel is not None:
                    with torch.no_grad():
                        self.model.eval()
                        val_loss, val_outputs, recall, specificity, auc, auprc = self.test(vx1, vx2, vlabel)
                        val_losses_epoch.append(val_loss)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop_count = 0
                            torch.save(self.model.state_dict(), checkpoint)
                        else:
                            early_stop_count += 1
                    self.scheduler.step(val_loss)
                else:
                    torch.save(self.model.state_dict(), checkpoint)

        figure_dir = os.path.join(figure_dir, 'finetune_part')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        plt.plot(train_losses_epoch, label='Training loss')
        plt.plot(val_losses_epoch, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(figure_dir, 'split' + str(self.rep) + '_fold' + str(self.i) + 'losses.png'))
        plt.clf()
        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        
        _, _, train_recall, train_specificity, train_roc_auc, train_auprc = self.test(x1, x2, label)

        _, _, val_recall, val_specificity, val_roc_auc, val_auprc = self.test(vx1, vx2, vlabel)

        _, _, test_recall, test_specificity, test_roc_auc, test_auprc = self.test(tx1, tx2, tlabel)

        return train_recall, val_recall, test_recall, train_specificity, val_specificity, test_specificity, train_roc_auc, val_roc_auc, test_roc_auc, train_auprc, val_auprc, test_auprc

    def test(self, x1, x2, label):
        with torch.no_grad():
            self.model.eval() 
            data_size = x1.size(0) 
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)),batch_size = self.batch_size, drop_last=False))
            losses = [] 
            preds = [] 
            labels = []
            probabilities_all = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                batch_label = label[batch_idx, :]
                fusion_out1, fusion_out2, pred_logits = self.model(batch_x1, batch_x2)
                
                probabilities = torch.sigmoid(pred_logits)
                predictions = (probabilities >= 0.5).int()
                loss = self.losses['loss'](pred_logits, batch_label)
                losses.append(loss.item()) 
                preds.extend(predictions.cpu().detach().numpy())
                labels.extend(batch_label.cpu().detach().numpy())
                probabilities_all.extend(probabilities.cpu().detach().numpy())

            recall = sklearn.metrics.recall_score(labels, preds, pos_label=1)
            roc_auc = sklearn.metrics.roc_auc_score(labels, probabilities_all)

            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, preds).ravel()
            # Calculate specificity
            specificity = tn / (tn + fp)

            # Calculate precision-recall pairs
            precision, recall_for_auprc, thresholds = sklearn.metrics.precision_recall_curve(labels, preds, pos_label=1)

            # Calculate AUPRC
            auprc = sklearn.metrics.auc(recall_for_auprc, precision)


            return np.mean(losses), pred_logits, recall, specificity, roc_auc, auprc
    