import torch
import numpy as np
import os 
from utils import z_score_norm, load_data
from model import DeepFusion, DownstreamPredictionHead
from objectives import cca_loss, cosine_similarity_loss, cross_entropy_loss
from train import Fusion_Solver, Downstream_Solver

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

torch.set_default_tensor_type(torch.DoubleTensor)
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_features', type = str, default = None, help = 'parent path that contains both Pyradiomics and clinical features')
parser.add_argument('--fusion_method', type = str, default = None, help = 'Fusion method of the following choices: concatenation, tensor, cosine_similarity, dcca, cross_attention')

args = parser.parse_args()

encoder_name = 'mlp'
input_feature_csv_path = args.path_to_features
fusion_method = args.fusion_method

       
device = torch.device('cuda')
print(f"Using device: {device}")

use_all_singular_values = False

# ======== Not tuned parameters  ========= # 

epoch_num = 100
early_stopping_epoch = 10
layer_sizes_mlp = [16] # [2*fusion_outdim, 16, 1]
reg_par = 1e-4

# ======== Tuned parameters  ========= # 

outdim_size_list = [16] # the size of the new space learned by the model (number of the new features) # [16, 32, 64]
hidden_size_list = [64] #[64, 128, 256] 
learning_rate_list = [5e-4] # [5e-4, 1e-4, 5e-5]
batch_size_list = [16] #[16, 32]
dropout_list = [0, 0.25]

# ======== Repeated cross validation  ========= # 
n_splits = 5 
num_rep = 1

df_data1, df_data2, df_label, pid_patient_level, label_patient_level = load_data(input_feature_csv_path)

print('Done reading in data')
# size of the input for view 1 and view 2 (number of features)
df_features1 = df_data1.iloc[:, (df_data1.columns.get_loc('diagnosis_nodule') + 1):]
df_features2 = df_data2.iloc[:, (df_data2.columns.get_loc('diagnosis_nodule') + 1):]

input_shape1 = df_features1.shape[1]
input_shape2 = df_features2.shape[1]

model_count = 0
df_all_results = pd.DataFrame()

for outdim_idx in range(len(outdim_size_list)):
    for hidden_idx in range(len(hidden_size_list)):
        for lr_idx in range(len(learning_rate_list)):
            for bs_idx in range(len(batch_size_list)):
                for dropout_idx in range(len(dropout_list)):
                    print('Model ', model_count)
                    
                    exp = os.path.join(os.getcwd(), fusion_method, 'saved_models_' + encoder_name, 'model_' + str(model_count))
                    figure_dir = os.path.join(os.getcwd(), fusion_method, 'figures_' + encoder_name, 'model_' + str(model_count))

                    # Check if directory exists
                    if not os.path.exists(exp):
                        os.makedirs(exp)

                    # number of layers with nodes in each one
                    layer_sizes1 = [hidden_size_list[hidden_idx], outdim_size_list[outdim_idx]]
                    layer_sizes2 = [hidden_size_list[hidden_idx], outdim_size_list[outdim_idx]]

                    ## Repeated CV 
                    rep_cv_auprc_train = []
                    rep_cv_auprc_val = [] 
                    rep_cv_auprc_test = [] 

                    rep_cv_recall_train = []
                    rep_cv_recall_val = [] 
                    rep_cv_recall_test = [] 

                    rep_cv_specificity_train = []
                    rep_cv_specificity_val = [] 
                    rep_cv_specificity_test = [] 

                    rep_cv_auroc_train = []
                    rep_cv_auroc_val = [] 
                    rep_cv_auroc_test = [] 
                    for rep in range(num_rep): 
                        seed = rep 
                        torch.manual_seed(seed)

                        # Initialize 5-fold CV
                        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                        for i, (train_val_index, test_index) in enumerate(cv_strategy.split(pid_patient_level, label_patient_level)):

                            pid_train_val, pid_test = pid_patient_level[train_val_index], pid_patient_level[test_index]
                            label_train_val, label_test = label_patient_level[train_val_index], label_patient_level[test_index]
                            
                            # Further split into train and val 
                            pid_train, pid_val, label_train, label_val = train_test_split(
                                    pid_train_val, 
                                    label_train_val,
                                    test_size=0.25, 
                                    random_state=seed, 
                                    stratify=label_train_val)
                            
                            # Map back to the original DataFrame to get tumor-level splits
                            train1 = df_data1[df_data1['pid'].isin(pid_train)]
                            train2 = df_data2[df_data2['pid'].isin(pid_train)]
                            val1 = df_data1[df_data1['pid'].isin(pid_val)]
                            val2 = df_data2[df_data2['pid'].isin(pid_val)]
                            test1 = df_data1[df_data1['pid'].isin(pid_test)]
                            test2 = df_data2[df_data2['pid'].isin(pid_test)]
                            status_train = df_label[df_label['pid'].isin(pid_train)]
                            status_val = df_label[df_label['pid'].isin(pid_val)]
                            status_test = df_label[df_label['pid'].isin(pid_test)]

                            train1 = torch.FloatTensor(train1.drop(columns=['pid', 'image_path', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)
                            train2 = torch.FloatTensor(train2.drop(columns=['pid', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)
                            val1 = torch.FloatTensor(val1.drop(columns=['pid', 'image_path', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)
                            val2 = torch.FloatTensor(val2.drop(columns=['pid', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)
                            test1 = torch.FloatTensor(test1.drop(columns=['pid', 'image_path', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)
                            test2 = torch.FloatTensor(test2.drop(columns=['pid', 'nodule_path', 'diagnosis_nodule'], errors='ignore').values)

                            train_label = torch.FloatTensor(status_train['diagnosis_nodule'].values)
                            val_label = torch.FloatTensor(status_val['diagnosis_nodule'].values)
                            test_label = torch.FloatTensor(status_test['diagnosis_nodule'].values)

                            # Normalize the features
                            train1, train2, val1, val2, test1, test2 = z_score_norm(train1, train2, val1, val2, test1, test2)
                        
                            train1, val1, test1 = train1.cuda().double(), val1.cuda().double(), test1.cuda().double()
                            train2, val2, test2 = train2.cuda().double(), val2.cuda().double(), test2.cuda().double()
                            train_label, val_label, test_label = train_label.unsqueeze(1).cuda().double(), val_label.unsqueeze(1).cuda().double(), test_label.unsqueeze(1).cuda().double()
                            
                            # Define loss functions 
                            corr_loss = cca_loss(outdim_size_list[outdim_idx], use_all_singular_values, device=device)
                            cosine_loss = cosine_similarity_loss()
                            ce_loss = cross_entropy_loss()

                            # Train DCCA or cosine similarity
                            if fusion_method == 'concatenation' or fusion_method == 'tensor':
                                fusion_model = DeepFusion(encoder_name, layer_sizes1, layer_sizes2, input_shape1,
                                            input_shape2, outdim_size_list[outdim_idx], dropout_list[dropout_idx], use_all_singular_values, device=device).double()
                            elif fusion_method == 'dcca':
                                fusion_model = DeepFusion(encoder_name, layer_sizes1, layer_sizes2, input_shape1,
                                            input_shape2, outdim_size_list[outdim_idx], dropout_list[dropout_idx], use_all_singular_values, device=device).double()
                                fusion_solver = Fusion_Solver(rep, i, fusion_model, corr_loss.loss,  outdim_size_list[outdim_idx], epoch_num, batch_size_list[bs_idx],
                                            learning_rate_list[lr_idx], reg_par, early_stopping_epoch, device=device)
                                fusion_model = fusion_solver.fit(train1, train2, val1, val2, test1, test2, figure_dir, checkpoint=os.path.join(exp, f'fusion_checkpoint_{rep}_{i}.model'))
                              
                            elif fusion_method == 'cosine_similarity':
                                fusion_model = DeepFusion(encoder_name, layer_sizes1, layer_sizes2, input_shape1,
                                            input_shape2, outdim_size_list[outdim_idx], dropout_list[dropout_idx], use_all_singular_values, device=device).double()
                                fusion_solver = Fusion_Solver(rep, i, fusion_model, cosine_loss.loss, outdim_size_list[outdim_idx], epoch_num, batch_size_list[bs_idx],
                                        learning_rate_list[lr_idx], reg_par, early_stopping_epoch, device=device)
                                fusion_model = fusion_solver.fit(train1, train2, val1, val2, test1, test2, figure_dir, checkpoint=os.path.join(exp, f'cosine_checkpoint_{rep}_{i}.model'))

                            # Finetune  
                            if fusion_method == 'tensor':
                                model = DownstreamPredictionHead(fusion_method, fusion_model, layer_sizes_mlp, outdim_size_list[outdim_idx]*outdim_size_list[outdim_idx], 1, dropout_list[dropout_idx], device=device)
                            else:
                                model = DownstreamPredictionHead(fusion_method, fusion_model, layer_sizes_mlp, 2*outdim_size_list[outdim_idx], 1, dropout_list[dropout_idx], device=device)
                            total_params = sum(p.numel() for p in model.parameters())
                            # print('total_params ', total_params)
                            # exit()
                            model_solver = Downstream_Solver(rep, i, model, {'loss': ce_loss.loss}, epoch_num, batch_size_list[bs_idx], learning_rate_list[lr_idx], reg_par, early_stopping_epoch, device=device)
                            train_recall, val_recall, test_recall, train_specificity, val_specificity, test_specificity, train_auroc, val_auroc, test_auroc, train_auprc, val_auprc, test_auprc  = model_solver.finetune(train1, train2, train_label, 
                                            val1, val2, val_label, 
                                            test1, test2, test_label, 
                                            figure_dir,
                                            checkpoint=os.path.join(exp, f'checkpoint_{rep}_{i}.model'))

                            # Track metrics 

                            rep_cv_recall_train.append(train_recall)
                            rep_cv_recall_val.append(val_recall)
                            rep_cv_recall_test.append(test_recall)

                            rep_cv_specificity_train.append(train_specificity)
                            rep_cv_specificity_val.append(val_specificity)
                            rep_cv_specificity_test.append(test_specificity)

                            rep_cv_auroc_train.append(train_auroc)
                            rep_cv_auroc_val.append(val_auroc)
                            rep_cv_auroc_test.append(test_auroc)

                            rep_cv_auprc_train.append(train_auprc)
                            rep_cv_auprc_val.append(val_auprc)
                            rep_cv_auprc_test.append(test_auprc)

                    print(f"Test auroc: {np.mean(rep_cv_auroc_test):.4f} ({np.std(rep_cv_auroc_test):.4f})")
                    print(f"Test auprc: {np.mean(rep_cv_auprc_test):.4f} ({np.std(rep_cv_auprc_test):.4f})")

                    df_train = pd.DataFrame({
                                        'all_fold_train_recall': [rep_cv_recall_train], 
                                        'average_train_recall': np.mean(rep_cv_recall_train), 
                                        'SD_train_recall': np.std(rep_cv_recall_train),
                                        
                                        'all_fold_train_specificity': [rep_cv_specificity_train], 
                                        'average_train_specificity': np.mean(rep_cv_specificity_train), 
                                        'SD_train_specificity': np.std(rep_cv_specificity_train),

                                        'all_fold_train_auroc': [rep_cv_auroc_train], 
                                        'average_train_auroc': np.mean(rep_cv_auroc_train), 
                                        'SD_train_auroc': np.std(rep_cv_auroc_train),

                                        'all_fold_train_auprc': [rep_cv_auprc_train], 
                                        'average_train_auprc': np.mean(rep_cv_auprc_train), 
                                        'SD_train_auprc': np.std(rep_cv_auprc_train)}
                                        )
                
                    df_val = pd.DataFrame({
                                            'all_fold_val_recall': [rep_cv_recall_val], 
                                            'average_val_recall': np.mean(rep_cv_recall_val), 
                                            'SD_val_recall': np.std(rep_cv_recall_val),
                                            
                                            'all_fold_val_specificity': [rep_cv_specificity_val], 
                                            'average_val_specificity': np.mean(rep_cv_specificity_val), 
                                            'SD_val_specificity': np.std(rep_cv_specificity_val),

                                            'all_fold_val_auroc': [rep_cv_auroc_val], 
                                            'average_val_auroc': np.mean(rep_cv_auroc_val), 
                                            'SD_val_auroc': np.std(rep_cv_auroc_val),
                                            
                                            'all_fold_val_auprc': [rep_cv_auprc_val], 
                                            'average_val_auprc': np.mean(rep_cv_auprc_val), 
                                            'SD_val_auprc': np.std(rep_cv_auprc_val)}
                                            )
                    df_test = pd.DataFrame({
                                            'all_fold_test_recall': [rep_cv_recall_test], 
                                            'average_test_recall': np.mean(rep_cv_recall_test), 
                                            'SD_test_recall': np.std(rep_cv_recall_test),
                                            
                                            'all_fold_test_specificity': [rep_cv_specificity_test], 
                                            'average_test_specificity': np.mean(rep_cv_specificity_test), 
                                            'SD_test_specificity': np.std(rep_cv_specificity_test),

                                            'all_fold_test_auroc': [rep_cv_auroc_test], 
                                            'average_test_auroc': np.mean(rep_cv_auroc_test), 
                                            'SD_test_auroc': np.std(rep_cv_auroc_test),
                                            
                                            'all_fold_test_auprc': [rep_cv_auprc_test], 
                                            'average_test_auprc': np.mean(rep_cv_auprc_test), 
                                            'SD_test_auprc': np.std(rep_cv_auprc_test)}
                                            )
                    df_params = pd.DataFrame({'model_index': [model_count], 
                                                'batch_size' : [batch_size_list[bs_idx]], 
                                                'hidden_size' : [hidden_size_list[hidden_idx]],
                                                'outdim_size' : [outdim_size_list[outdim_idx]],
                                                'lr' : [learning_rate_list[lr_idx]],
                                                'mlp_dropout' : [dropout_list[dropout_idx]]
                                                })
                    df_this_results = pd.concat([df_params, df_train, df_val, df_test], axis = 1)
                    df_all_results = pd.concat([df_all_results, df_this_results])
                
                    df_all_results.to_csv(os.path.join(os.getcwd(),  fusion_method,  'gs_results_' + encoder_name + '.csv'), index = None)
                    model_count += 1

