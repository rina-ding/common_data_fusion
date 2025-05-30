import pandas as pd
import os

def z_score_norm(train1, train2, val1, val2, test1, test2):
    train1_mean = train1.mean()
    train1_sd = train1.std()
    train1 = (train1 - train1_mean) / train1_sd
    val1 = (val1 - train1_mean) / train1_sd
    test1 = (test1 - train1_mean) / train1_sd

    train2_mean = train2.mean()
    train2_sd = train2.std()
    train2 = (train2 - train2_mean) / train2_sd
    val2 = (val2 - train2_mean) / train2_sd
    test2 = (test2 - train2_mean) / train2_sd
    return train1, train2, val1, val2, test1, test2

def min_max_norm(train1, train2, val1, val2, test1, test2):
    train1_max = train1.max()
    train1_min = train1.min()
    train1 = (train1 - train1_min) / (train1_max - train1_min)
    val1 = (val1 - train1_min) / (train1_max - train1_min)
    test1 = (test1 - train1_min) / (train1_max - train1_min)

    train2_max = train2.max()
    train2_min = train2.min()
    train2 = (train2 - train2_min) / (train2_max - train2_min)
    val2 = (val2 - train2_min) / (train2_max - train2_min)
    test2 = (test2 - train2_min) / (train2_max - train2_min)
    return train1, train2, val1, val2, test1, test2

def load_data(data_dir_path): 
    nlst_ct_df = pd.read_csv(os.path.join(data_dir_path, 'extracted_pyradiomics.csv'))
    clinical_df = pd.read_csv(os.path.join(data_dir_path, 'clinical_features.csv'))
    label_df = clinical_df[['pid', 'nodule_path', 'diagnosis_nodule']]

    pid_patient_level = clinical_df.drop_duplicates(subset = ['pid'])['pid'].values
    status_patient_level = clinical_df.drop_duplicates(subset = ['pid'])['diagnosis_nodule'].values

    return nlst_ct_df, clinical_df, label_df, pid_patient_level, status_patient_level