import numpy as np
import os 
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchvision.transforms.functional as TF
import random

class MultimodalDataProcessor(Dataset):
    def __init__(self, df, to_transform, encoder_name, fusion_method, roi_size, resampled_spacing):
        self.df = df
        self.to_transform = to_transform
        self.encoder_name = encoder_name
        self.fusion_method = fusion_method
        self.roi_size = roi_size
        self.resampled_spacing = resampled_spacing
    
    def image_resample(self, input_image, is_label=False):
        resample_filter = sitk.ResampleImageFilter()

        input_spacing = input_image.GetSpacing()
        input_direction = input_image.GetDirection()
        input_origin = input_image.GetOrigin()
        input_size = input_image.GetSize()

        output_spacing = self.resampled_spacing
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
    
    def resample_image(self, img):
        # Load image and (optionally) mask
        spacing = np.asarray(img.GetSpacing())
        iso_voxel_size = 1

        # Resample image
        img = self.image_resample(img, is_label=False)
        return img
    
    def resample_mask(self, mask):
        mask = self.image_resample(mask, is_label=True)
        return mask
    
    def extract_roi_from_centroid(self, image, x, y, z):
        roi_size = self.roi_size
     
        depth, height, width = image.shape
        
        # ROI size
        roi_depth, roi_height, roi_width = roi_size
        
        # Compute the start and end indices for slicing, ensuring bounds are valid
        z_start = max(z - roi_depth // 2, 0)
        z_end = min(z + roi_depth // 2, depth)
        
        x_start = max(x - roi_width // 2, 0)
        x_end = min(x + roi_width // 2, width)
        
        y_start = max(y - roi_height // 2, 0)
        y_end = min(y + roi_height // 2, height)
        
        # Slice the image to get the ROI

        roi = image[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad the ROI if it goes beyond image boundaries (using np.pad)
        pad_z = roi_depth - roi.shape[0]
        pad_y = roi_width - roi.shape[1]
        pad_x = roi_height - roi.shape[2]
        
        if pad_z > 0 or pad_x > 0 or pad_y > 0:
            roi = np.pad(roi, 
                        ((0, pad_z), (0, pad_y), (0, pad_x)), 
                        mode='constant', constant_values=0)
        
        return roi

    def load_and_process_image(self, img_file, x_pixel, y_pixel, z_pixel):
        sitk_image = sitk.ReadImage(img_file)

        sitk_image_resampled = self.resample_image(sitk_image)
        array_image = sitk.GetArrayFromImage(sitk_image_resampled)

        array_image = (array_image - array_image.min()) / (array_image.max() - array_image.min())

        array_roi = self.extract_roi_from_centroid(array_image, x_pixel, y_pixel, z_pixel)
       
        array_roi = np.expand_dims(array_roi, axis=3)
        array_roi = array_roi.transpose((3, 0, 1, 2))
        tensor_roi = torch.from_numpy(array_roi).type(torch.FloatTensor)

        return tensor_roi

    def transform_img(self, img):
       
        # Random horizontal flipping
        if random.random() > 0.6:
            img = TF.hflip(img)

        # Random vertical flipping
        if random.random() > 0.6:
            img = TF.vflip(img)
        
        return img

    def __getitem__(self, i):
        patient_id = self.df['pid'][i]
        x_pixel, y_pixel, z_pixel = self.df['pixel_X'][i], self.df['pixel_Y'][i], self.df['pixel_Z'][i]

        label = np.array([self.df['diagnosis_nodule'][i]])
        label = torch.from_numpy(label).float()

        t2w_img_file = self.df['m1_image_path'][i]
        t2w_processed = self.load_and_process_image(t2w_img_file, x_pixel, y_pixel, z_pixel)
        adc_img_file = self.df['m2_image_path'][i]
        adc_processed = self.load_and_process_image(adc_img_file, x_pixel, y_pixel, z_pixel)

        if self.to_transform:
            t2w_processed = self.transform_img(t2w_processed)
            adc_processed = self.transform_img(adc_processed)

        return {"m1_image": t2w_processed, 'm2_image': adc_processed, 'label':label, 'patient_id': patient_id}

    def __len__(self):
       return len(self.df)

