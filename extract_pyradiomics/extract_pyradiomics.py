import os
from radiomics import featureextractor 
import radiomics
import SimpleITK as sitk
import logging
import pandas as pd
import argparse

def get_pyradiomic_features(SITK_image, SITK_mask, append_header = False):
    log_file = './log_file.txt'
    handler = logging.FileHandler(filename=log_file, mode='w')  # overwrites log_files from previous runs. Change mode to 'a' to append.
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")  # format string for log messages
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    
    header, values = [], []
    settings = {}
    settings['binwidth'] = 25
    settings['label'] = 1
    settings['correctMask'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings, normalize = True)
    extractor.addProvenance(False)
    result = extractor.execute(SITK_image, SITK_mask)
    
    for key, value in result.items():
        if append_header:
            header.append(key)
        values.append(value)
    
    if append_header:
        return header, values
    else:
        return values

def main_save_features_one_lesion_3D_radiomic(patient_id, image_path, lesion_path, label):  
    header, all_feature_values = [], []
  
    sitk_image = sitk.ReadImage(image_path)
    sitk_mask = sitk.ReadImage(lesion_path)

    append_header = True
    header, features = get_pyradiomic_features(sitk_image, sitk_mask, append_header)
    
    features.insert(0, label)
    header.insert(0, 'diagnosis_nodule')
    features.insert(0, lesion_path)
    header.insert(0, 'nodule_path')
    features.insert(0, image_path)
    header.insert(0, 'image_path')
    features.insert(0, patient_id)
    header.insert(0, 'pid')
    all_feature_values.append(features)
    df_features = pd.DataFrame(columns = header, data = all_feature_values)

    return df_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type = str, default = None, help = 'path to the csv file that contains pid, image_path, nodule_path, and diagnosis_nodule')
    parser.add_argument('--path_to_extracted_features', type = str, default = None, help = 'path to the extracted features')

    args = parser.parse_args()

    path_to_csv = args.path_to_csv
    path_to_extracted_features = args.path_to_extracted_features
    radiomic_feature_list = []

    df_cases = pd.read_csv(path_to_csv)
    for i in range(len(df_cases)):
        patient_id = df_cases['pid'][i]
        img_file = df_cases['image_path'][i]
        annotation_file = df_cases['nodule_path'][i]
        label = df_cases['diagnosis_nodule'][i]
        print(i)
        print(patient_id)
        if 'radraid' in annotation_file:
            annotation_file = annotation_file.replace('/workspace/radraid', '/workspace/radraid/dataset')
        
        df_all_features = main_save_features_one_lesion_3D_radiomic(patient_id, img_file, annotation_file, label)
        radiomic_feature_list.append(df_all_features)

        df_radiomic = pd.concat(radiomic_feature_list)
        df_radiomic.to_csv(os.path.join(path_to_extracted_features, 'extracted_pyradiomics.csv'), index = None)
