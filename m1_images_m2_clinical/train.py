import torch
import numpy as np
import time
import os 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MultimodalDataProcessor
import sklearn.metrics 

class Fusion_Solver():
    def __init__(self, encoder_name, fusion_method, i, fusion_model, loss, outdim_size, epoch_num, batch_size, learning_rate, reg_par, roi_size, resampled_spacing, early_stopping_epoch, device=torch.device('cpu')):
        self.encoder_name = encoder_name
        self.fusion_method = fusion_method
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

        self.roi_size = roi_size
        self.resampled_spacing = resampled_spacing
        self.early_stopping_epoch = early_stopping_epoch

    def fit(self, train_patients, val_patients=None, test_patients=None, figure_dir = "", checkpoint='fusion_checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # Set up Dataset
        train_dataset = MultimodalDataProcessor(train_patients, True, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)
        val_dataset = MultimodalDataProcessor(val_patients, False, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)
        test_dataset = MultimodalDataProcessor(test_patients, False, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)

        # Set up Dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,  shuffle=True, drop_last=False, num_workers = 4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 4)

        if val_patients is not None:
            best_val_loss = np.inf

        train_losses_epoch = [] 
        val_losses_epoch = []
        running_val_loss = 0
        early_stop_count = 0
        for epoch in range(self.epoch_num):
            if early_stop_count < self.early_stopping_epoch:
                train_losses = []
                print(f"Fusion epoch - {epoch}/{self.epoch_num}")
                self.model.train()
                for index, data in enumerate(train_loader):
                    ct_image, semantic, label, patient_id = data['ct_image'].to(self.device, dtype=torch.float), data['semantic'].to(self.device, dtype=torch.int), data['label'].to(self.device, dtype=torch.float), data['patient_id']
                    self.optimizer.zero_grad()
                    o1, o2 = self.model(ct_image, semantic)
                    loss = self.loss(o1, o2)
                    train_losses.append(loss.item())
                    loss.backward()
                   
                    self.optimizer.step()
                train_loss = np.mean(train_losses)
                train_losses_epoch.append(train_loss) 

                if val_patients is not None:
                    with torch.no_grad():
                        self.model.eval()
                        val_loss, val_outputs = self.test(val_loader)
                        val_losses_epoch.append(val_loss)
                        running_val_loss += float(val_loss.item()) * len(val_loader)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop_count = 0
                            torch.save(self.model.state_dict(), checkpoint)
                        else:
                            early_stop_count += 1
                    
                    avg_val_loss = running_val_loss / len(val_loader)
                    self.scheduler.step(avg_val_loss)
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
        plt.savefig(os.path.join(figure_dir, 'split' + str(0) + '_fold' + str(self.i) + 'losses.png'))
        plt.clf()

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)

        if test_patients is not None:
            loss, output = self.test(test_loader)
        return self.model

    def test(self, test_loader):
        with torch.no_grad():
            losses, outputs = self._get_outputs(test_loader)
            return np.mean(losses), outputs

    def _get_outputs(self, test_loader):
        with torch.no_grad():
            self.model.eval()
           
            losses = []
            outputs1 = []
            outputs2 = []
            for index, data in enumerate(test_loader):
                ct_image, semantic, label, patient_id = data['ct_image'].to(self.device, dtype=torch.float), data['semantic'].to(self.device, dtype=torch.int), data['label'].to(self.device, dtype=torch.float), data['patient_id']

                o1, o2 = self.model(ct_image, semantic)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


class Downstream_Solver():
    def __init__(self, encoder_name, fusion_method, i, model, losses, epoch_num, batch_size, learning_rate, reg_par, roi_size, resampled_spacing, early_stopping_epoch, device=torch.device('cpu')):
        self.encoder_name = encoder_name
        self.fusion_method = fusion_method
        self.i = i
        self.device = device
        self.model = model
        self.model.to(self.device)
        
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.losses = losses 
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = reg_par)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

        self.roi_size = roi_size
        self.resampled_spacing = resampled_spacing
        self.early_stopping_epoch = early_stopping_epoch
        
    def finetune(self, 
            train_patients, 
            val_patients=None, 
            test_patients=None, 
            figure_dir = '',
            checkpoint='checkpoint.model'):
        # Set up Dataset
        train_dataset = MultimodalDataProcessor(train_patients, True, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)
        val_dataset = MultimodalDataProcessor(val_patients, False, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)
        test_dataset = MultimodalDataProcessor(test_patients, False, self.encoder_name, self.fusion_method, self.roi_size, self.resampled_spacing)

        print('train size ', len(train_dataset))
        print('val size ', len(val_dataset))
        print('test size ', len(test_dataset))

        # Set up Dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,  shuffle=True, drop_last=False, num_workers = 4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 4)

        if val_patients is not None:
            best_val_loss = np.inf

        train_losses_epoch = [] 
        val_losses_epoch = []
        running_val_loss = 0
        early_stop_count = 0

        for epoch in range(self.epoch_num):
            if early_stop_count < self.early_stopping_epoch:

                print('Finetune epoch ', epoch)
                train_losses = [] ##
                self.model.train()
                for index, data in enumerate(train_loader):
                    ct_image, semantic, label, patient_id = data['ct_image'].to(self.device, dtype=torch.float), data['semantic'].to(self.device, dtype=torch.int), data['label'].to(self.device, dtype=torch.float), data['patient_id']

                    self.optimizer.zero_grad()
                    _, _, pred_logits = self.model(ct_image, semantic)
                    loss = self.losses['loss'](pred_logits, label)
                    train_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                train_loss = np.mean(train_losses)
                train_losses_epoch.append(train_loss) 

                if val_patients is not None:
                    with torch.no_grad():
                        self.model.eval()
                        val_loss, val_outputs, recall, specificity, auc, auprc = self.test(val_loader)
                        val_losses_epoch.append(val_loss)
                        running_val_loss += float(val_loss.item()) * len(val_loader)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop_count = 0
                            torch.save(self.model.state_dict(), checkpoint)
                        else:
                            early_stop_count += 1
                    avg_val_loss = running_val_loss / len(val_loader)
                    self.scheduler.step(avg_val_loss)
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
        plt.savefig(os.path.join(figure_dir, 'split' + str(0) + '_fold' + str(self.i) + 'losses.png'))
        plt.clf()
        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        
        _, _, train_recall, train_specificity, train_roc_auc, train_auprc = self.test(train_loader)

        _, _, val_recall, val_specificity, val_roc_auc, val_auprc = self.test(val_loader)

        _, _, test_recall, test_specificity, test_roc_auc, test_auprc = self.test(test_loader)

        return train_recall, val_recall, test_recall, train_specificity, val_specificity, test_specificity, train_roc_auc, val_roc_auc, test_roc_auc, train_auprc, val_auprc, test_auprc
    
    def test(self, test_loader):
        with torch.no_grad():
            self.model.eval() 
            losses = [] 
            preds = [] 
            labels = []
            probabilities_all = []

            for index, data in enumerate(test_loader):
                ct_image, semantic, label, patient_id = data['ct_image'].to(self.device, dtype=torch.float), data['semantic'].to(self.device, dtype=torch.int), data['label'].to(self.device, dtype=torch.float), data['patient_id']

                _, _, pred_logits = self.model(ct_image, semantic)
                probabilities = torch.sigmoid(pred_logits)
                probabilities_all.extend(probabilities.cpu().detach().numpy())

                predictions = (probabilities >= 0.5).int()
                loss = self.losses['loss'](pred_logits, label)
                losses.append(loss.item()) 
                preds.extend(predictions.cpu().detach().numpy())
                labels.extend(label.cpu().detach().numpy())
                
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
     