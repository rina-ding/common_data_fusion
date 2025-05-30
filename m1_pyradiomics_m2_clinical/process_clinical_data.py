
import os 

from glob import glob
import shutil
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df_nlst = pd.read_csv('/workspace/lottie_clip_data/nlst.csv')

# One hot encode semantic features

df_semantic = df_nlst[['nodule_margin_conspicuity', 'nodule_margins','additional_nodule_margins', 
                       'nodule_shape', 'nodule_consistency', 'nodule_reticulation', 'cyst-like_spaces', 
                       'intra-nodular_bronchiectasis', 'necrosis', 'cavitation', 'eccentric_calcification', 
                       'airway_cut-off', 'pleural_attachment','pleural_retraction',
                         'vascular_convergence', 'septal_stretching', 'paracicatricial_emphysema',
                         'predominant_nature_of_lung_parenchyma', 'emphysema_presence', 'emphysema_type', 
                         'emphysema_distribution', 'fibrosis', 'lymphadenopathy']]


imputer = SimpleImputer(strategy="most_frequent")
df_imputed = pd.DataFrame(imputer.fit_transform(df_semantic), columns=df_semantic.columns)

# Step 2: Perform one-hot encoding on categorical columns
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(df_imputed)

# Step 3: Convert encoded data back to a DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(df_semantic.columns))

df_final = pd.concat([df_nlst[['pid', 'image_path', 'nodule_path', 'diagnosis_nodule']], encoded_df], axis = 1)
# df_final = df_final.rename(columns = {'diagnosis_nodule':'label', 'nodule_path':'lesion_path'})
print(df_final)
df_final.to_csv('/workspace/lottie_clip_data/nlst_semantic_encoded.csv', index = None)
