import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

import os 
from model import DeepFusion, DownstreamPredictionHead
from objectives import cca_loss, cosine_similarity_loss, cross_entropy_loss
from train import Fusion_Solver, Downstream_Solver 
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_image_data_csv', type = str, default = None, help = 'path to the csv file that contains information about the image data')
parser.add_argument('--fusion_method', type = str, default = None, help = 'Fusion method of the following choices: concatenation, tensor, cosine_similarity, dcca, cross_attention')

args = parser.parse_args()
csv_path = args.path_to_image_data_csv
fusion_method = args.fusion_method

# Suppress all warnings
warnings.filterwarnings("ignore")
seed = 1234
device = torch.device('cuda')
print(f"Using device: {device}")
use_all_singular_values = False
encoder_name = 'vit' 

# ======== Not tuned parameters  ========= # 
epoch_num = 100
layer_sizes_mlp = [16] # [2*fusion_outdim, 16, 1]
reg_par = 1e-4
early_stopping_epoch = 10
roi_size = [16, 64, 64]
resampled_spacing = [0.66, 0.66, 1.5]

# ======== Tuned parameters  ========= # 
outdim_size_list = [32] # the size of the new space learned by the model (number of the new features)
learning_rate_list = [1e-4]
batch_size_list = [16]
dropout_list = [0]

df_all_patients = pd.read_csv(csv_path)

df_all_patients = df_all_patients.drop_duplicates(subset = ['nodule_path']).reset_index(drop = True) # Somehow there are some duplicated nodule rows like /workspace/whsu/Ticket8859/AnnotationPlusImage/10076_105340/2000-01-02/CT-Chest-at-TLC-Supine-1.2.276.0.7230010.3.1.3.898003985.9612.1506715033.236/Contours/NIFTI/Lesion_1000878719.nii.gz
df_patient_level = df_all_patients.drop_duplicates(subset = ['pid'])
df_patient_level = df_patient_level.reset_index(drop = True)
model_count = 0
df_all_results = pd.DataFrame()

for outdim_idx in range(len(outdim_size_list)):
    for lr_idx in range(len(learning_rate_list)):
        for bs_idx in range(len(batch_size_list)):
            for dropout_idx in range(len(dropout_list)):

                print('Model ', model_count)
                exp = os.path.join(os.getcwd(), fusion_method, 'saved_models_' + encoder_name, 'model_' + str(model_count))
                figure_dir = os.path.join(os.getcwd(), fusion_method, 'figures_' + encoder_name, 'model_' + str(model_count))

                # Check if directory exists
                if not os.path.exists(exp):
                    os.makedirs(exp)

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

                seed = 1
                torch.manual_seed(seed)

                # Initialize 5-fold CV
                n_splits = 5 
                cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for i, (train_val_index, test_index) in enumerate(cv_strategy.split(df_patient_level, df_patient_level['diagnosis_nodule'])):

                    df_patients_train_val, df_test_patients = df_patient_level.iloc[train_val_index, :], df_patient_level.iloc[test_index, :]
                    df_patients_train_val = df_patients_train_val.reset_index(drop = True)
                    df_test_patients = df_test_patients.reset_index(drop = True)

                    # Further split into train and val 
                    df_train_patients, df_val_patients = train_test_split(
                            df_patients_train_val,
                            test_size=0.25, 
                            random_state=seed, 
                            stratify=df_patients_train_val['diagnosis_nodule'])

                    df_train_patients = df_train_patients.reset_index(drop = True)
                    df_val_patients = df_val_patients.reset_index(drop = True)

                    df_train_lesions = df_all_patients[df_all_patients['pid'].isin(df_train_patients['pid'])].reset_index(drop = True)
                    df_val_lesions = df_all_patients[df_all_patients['pid'].isin(df_val_patients['pid'])].reset_index(drop = True)
                    df_test_lesions = df_all_patients[df_all_patients['pid'].isin(df_test_patients['pid'])].reset_index(drop = True)

                    # Define loss functions 
                    corr_loss = cca_loss(outdim_size_list[outdim_idx], use_all_singular_values, device=device)
                    cosine_loss = cosine_similarity_loss()
                    ce_loss = cross_entropy_loss()

                    # Train DCCA or cosine similarity
                    if fusion_method == 'concatenation' or fusion_method == 'tensor' or fusion_method == 'cross_attention':
                        fusion_model = DeepFusion(outdim_size_list[outdim_idx], encoder_name, fusion_method, roi_size, device)
                    elif fusion_method == 'dcca':
                        fusion_model = DeepFusion(outdim_size_list[outdim_idx], encoder_name, fusion_method, roi_size, device)
                        fusion_solver = Fusion_Solver(encoder_name, fusion_method, i, fusion_model, corr_loss.loss, outdim_size_list[outdim_idx], epoch_num, batch_size_list[bs_idx],
                                learning_rate_list[lr_idx], reg_par, roi_size, resampled_spacing, early_stopping_epoch, device=device)
                        fusion_model = fusion_solver.fit(df_train_lesions, df_val_lesions, df_test_lesions, figure_dir, checkpoint=os.path.join(exp, f'fusion_checkpoint_{0}_{i}.model'))
                    elif fusion_method == 'cosine_similarity':
                        fusion_model = DeepFusion(outdim_size_list[outdim_idx], encoder_name, fusion_method, roi_size, device)
                        fusion_solver = Fusion_Solver(encoder_name, fusion_method, i, fusion_model, cosine_loss.loss, outdim_size_list[outdim_idx], epoch_num, batch_size_list[bs_idx],
                                learning_rate_list[lr_idx], reg_par, roi_size, resampled_spacing, early_stopping_epoch, device=device)
                        fusion_model = fusion_solver.fit(df_train_lesions, df_val_lesions, df_test_lesions, figure_dir, checkpoint=os.path.join(exp, f'fusion_checkpoint_{0}_{i}.model'))
                    
                    # Finetune  
                    if fusion_method == 'tensor':
                        model = DownstreamPredictionHead(fusion_method, fusion_model, layer_sizes_mlp, outdim_size_list[outdim_idx]*outdim_size_list[outdim_idx], 1, dropout_list[dropout_idx], device=device)
                    else:
                        model = DownstreamPredictionHead(fusion_method, fusion_model, layer_sizes_mlp, 2*outdim_size_list[outdim_idx], 1, dropout_list[dropout_idx], device=device)
                    total_params = sum(p.numel() for p in model.parameters())
                    # print('total_params ', total_params)
                    model_solver = Downstream_Solver(encoder_name, fusion_method, i, model, {'loss': ce_loss.loss}, epoch_num, batch_size_list[bs_idx], learning_rate_list[lr_idx], reg_par, roi_size, resampled_spacing, early_stopping_epoch, device=device)
                    train_recall, val_recall, test_recall, train_specificity, val_specificity, test_specificity, train_auroc, val_auroc, test_auroc, train_auprc, val_auprc, test_auprc  = model_solver.finetune(df_train_lesions, 
                                    df_val_lesions, 
                                    df_test_lesions, 
                                    figure_dir,
                                    checkpoint=os.path.join(exp, f'checkpoint_{0}_{i}.model'))


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
                            'hidden_size' : [outdim_size_list[outdim_idx]],
                            'lr' : [learning_rate_list[lr_idx]],
                            'dropout' : [dropout_list[dropout_idx]]
                            })
                df_this_results = pd.concat([df_params, df_train, df_val, df_test], axis = 1)
                df_all_results = pd.concat([df_all_results, df_this_results], axis = 0)
                
                df_all_results.to_csv(os.path.join(os.getcwd(), fusion_method, 'gs_results_' + encoder_name + '.csv'), index = None)

                model_count += 1