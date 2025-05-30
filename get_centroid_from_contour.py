import os
import sys
from glob import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

import logging
from shutil import copy
import pandas as pd
from PIL import Image 
from scipy.ndimage import center_of_mass

def get_centroids(mask):
    """
    Efficiently calculates the centroid of a binary mask.
    """
    if np.count_nonzero(mask) == 0:
        return 0, 0, 0  # Handle empty mask gracefully

    # Use center_of_mass directly on the binary mask
    mask = (mask > 0).astype(np.float32)
    centroid = center_of_mass(mask)
    return int(centroid[0]), int(centroid[1]), int(centroid[2]) # z, y, x

def image_resample(input_image, is_label=False):
    resample_filter = sitk.ResampleImageFilter()

    input_spacing = input_image.GetSpacing()
    input_direction = input_image.GetDirection()
    input_origin = input_image.GetOrigin()
    input_size = input_image.GetSize()

    output_spacing = resampled_spacing
    output_origin = input_origin
    output_direction = input_direction
    output_size = np.ceil(np.asarray(input_size) * np.asarray(input_spacing) / np.asarray(output_spacing)).astype(int)

    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetSize(output_size.tolist())
    resample_filter.SetOutputDirection(output_direction)
    if is_label:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resample_filter.Execute(input_image)
    else:
        resample_filter.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample_filter.Execute(input_image)

    return resampled_image

def resample_mask(mask):
    mask = image_resample(mask, is_label=True)
    return mask

df_cases = pd.read_csv('/workspace/modeling/prostate_data_fusion/ucla_cspca/feb_2025_ucla_cases_with_labels.csv')
resampled_spacing = [0.5, 0.5, 3] 
root_path = '/workspace/prostate_data/cleaned_cases_with_annotations_feb_2025'
insert_index = df_cases.columns.get_loc('csPCa') 

df_cases.insert(insert_index+1, 'pixel_X', '')
df_cases.insert(insert_index+2, 'pixel_Y', '')
df_cases.insert(insert_index+3, 'pixel_Z', '')
for i in range(len(df_cases)):
    print(i)
    annotation_file = df_cases['prostate_path'][i]
    
    sitk_mask = sitk.ReadImage(annotation_file)

    sitk_mask_resampled = resample_mask(sitk_mask)
    array_mask = sitk.GetArrayFromImage(sitk_mask_resampled)
    z, y, x = get_centroids(array_mask)

    df_cases['pixel_X'][i] = x
    df_cases['pixel_Y'][i] = y
    df_cases['pixel_Z'][i] = z

df_cases.to_csv('/workspace/modeling/prostate_data_fusion/ucla_cspca/feb_2025_ucla_cases_with_labels_centroids.csv', index = None)
