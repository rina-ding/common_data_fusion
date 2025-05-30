# common_data_fusion

Notes:
1. This project was done on the DGX server (`/radraid/ruiwending`), so all the file paths are specific to that server. 
2. Binary classification as the prediction task.
3. All models were trained and evaluated using 5-fold CV, where in each fold, there are 60% train, 20% val, and 20% test data. Each model was tuned using grid search (Rina wrote simple for loops to achieve this). The best hyperparameter sets were chosen based on the average validation metric (e.g. AUPRC). The split was done at patient level.
4. The fusion methods implemented are concatenation, tensor product, deep canonical correlation analysis (CCA), cosine similarity, and cross-attention. For "When modality 1 is pyradiomics and modality 2 is clinical data in tabular format", cross-attention was not implemented since it is not natural to perform cross attention on non-tokenized data.
5. We assume that the users have nodule contour annotations for the imaging data.
6. Note that when you use deep CCA as the fusion method, you might get errors when computing the CCA loss due numericial instability. Rina tried to adjust some CCA loss related parameters but a lot of times, the program will still fail. In that case, you need to skip that run, unless you can find a solution to debug.
7. For cross attention fusion, the code assumes you do two-way attention instead of only one-way.
8. In the future, if needed for any reason, the exact (uncleaned) code that Rina used for the prostate BCR prediction is under `/radraid/ruiwending/modeling/prostate_data_fusion/ucla_bcr`, for prostate csPCa classification is under `/radraid/ruiwending/modeling/prostate_data_fusion/ucla_cspca`, for prostate PIRADS classification is under `/radraid/ruiwending/modeling/prostate_data_fusion/ucla_pirads`, for lung malignancy classification is under `/radraid/ruiwending/modeling/lung_data_fusion/nlst_diagnosis`.


## Rina's writeup and slides
- The writeup of this project is in Chapter 7 of Rina's dissertation [here](https://drive.google.com/file/d/1Qw86mGwi1_jRhEJ48_p2_1kQkTV1C913/view?usp=sharing)
- Slides are [here](https://docs.google.com/presentation/d/1p2YAfoOuo9b5kHmZUfxrX1CXjVRkiiq0hQ7f5XMdft4/edit?usp=sharing)

## Required packages
First, create a pytorch docker container using:
```
docker run --shm-size=2g --user $(id -u):$(id -g) -it --gpus all --rm  -v /radraid/<YOUR USER NAME>:/workspace -v /radraid/whsu:/workspace/whsu -v /radraid/luotingzhuang:/workspace/radraid -v /etc/localtime:/etc/localtime:ro rddocker:latest
```

`<YOUR USER NAME>` is your username folder on DGX, such as `/radraid/ruiwending`

`/radraid/whsu:/workspace/whsu` and `/radraid/luotingzhuang:/workspace/radraid` are necessary since all the NLST lung CT datasets are under those paths.

`rddocker:latest` is the docker to be used, which was created by Rina under `/radraid/ruiwending/rddocker`

Then install all packages by running the following commands:

```
cd data_fusion_pipeline
```

```
chmod +x pip_commands.sh
```
```
./pip_commands.sh
```

## When modality 1 is pyradiomics and modality 2 is clinical data in tabular format

### Extract pyradiomics

```
cd extract_pyradiomics
```

```
python extract_pyradiomics.py --path_to_csv ../nlst_toy.csv --path_to_extracted_features ./features
```

The input csv file should contain the following columns in the following order:

`pid`: patient/case ID

`image_path`: path to the image scan in `.nii.gz` format

`nodule_path`: path to the nodule contour annotation in `.nii.gz` format

`diagnosis_nodule`: nodule malignancy label, with 1 being maglignant, 0 being non-malignant 

### Extract clinical

Please one-hot encode each clinical variable (assuming all your features are categorical) and put the csv file under the same folder as in the extracted pyradiomic features from the last step. The processed file should have the following columns in the following order:

`pid`: patient/case ID

`nodule_path`: path to the nodule contour annotation in `.nii.gz` format

`diagnosis_nodule`: nodule malignancy label, with 1 being maglignant, 0 being non-malignant 

And the rest of the columns are the one-hot encoded features.

For your reference, Rina used `./m1_pyradiomics_m2_clinical/process_clinical_data.py` to generate one-hot encoded data for NLST.

### Run model

Modality 1 and 2's encoders are both MLP. This script automatically runs 4 fusion methods including 'concatenation', 'tensor', 'dcca', 'cosine_similarity'.

```
cd m1_pyradiomics_m2_clinical
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_features ../extract_pyradiomics/features --fusion_method <fusion_method>
```

`path_to_features`: parent path that contains both Pyradiomics and clinical features

`fusion_method`: there are 4 choices available and you must use the exact same naming/spelling as them: concatenation, tensor, cosine_similarity, dcca. There's no cross attention since they require tokenized data but pyradiomics and clinical tabular data are not.

You can change the other hyperparameters to be tuned under the `Tuned parameters` section in `main.py`, by adding more values for each hyperparameter.  You can also change things like "number of epochs" under the `Not tuned parameters` section.

## When modality 1 is image data and modality 2 is clinical data in natural language format

### Image data input csv

The csv file should contain the following columns:

`pid`: patient/case ID

`image_path`: path to the image scan in `.nii.gz` format

`nodule_path`: path to the nodule contour annotation in `.nii.gz` format

`pixel_X`: coordinate X of the nodule's centroid, in pixel (instead of in real-world coordinates). 

`pixel_Y`: coordinate Y of the nodule's centroid, in pixel (instead of in real-world coordinates)

`pixel_Z`: coordinate Z of the nodule's centroid, in pixel (instead of in real-world coordinates)

`diagnosis_nodule`: nodule malignancy label, with 1 being maglignant, 0 being non-malignant 

`nodule_id`: if there are multiple timepoints for the imaging data, then this is the index of the nodule. 0 means timepoint 0, 1 means timepoint 1, and 2 means timepoint 2.

IMPORTANT: when generating `pixel_X`, `pixel_Y`, and `pixel_Z`, you need to use the same resampled voxel spacing as what you would use in your modeling script `main.py`. An example code for generating `pixel_X`, `pixel_Y`, and `pixel_Z` from a provided nodule mask and image scan can be found at [here](./get_centroid_from_contour.py)

### Clinical data converted to natural language texts

Please talk to Lottie if you have questions on how to generate the texts. 

The folder structure the text data should be:

```
├── report_generation
  ├── pid_noduleID.txt    
  ├── pid_noduleID.txt    
  ├── pid_noduleID.txt    
  ...

```

Please check the content and format of each txt file under `/radraid/ruiwending/lottie_clip_data/report_generation`. Rina's current code only works if you use the exact same format as these txt files.

### Download CLIP text transformer weights

Download from [here](https://drive.google.com/file/d/13_uYXuKZYtL6DxOFVZgIjwOWGISwoY_k/view?usp=sharing) and put the file under `./m1_images_m2_clinical/longclip_main`

### Run model

The image branch's encoder is a 3D ViT without pretraining (It's better to have a pretrained one if you are able to find one in the future), and the text branch's encoder is a text transformer pretrained from [LongCLIP](https://github.com/beichenzbc/Long-CLIP)

```
cd m1_images_m2_clinical
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_image_data_csv ../nlst_with_nodule_xyz_toy.csv --fusion_method <fusion_method>
```

`path_to_image_data_csv`: path to the csv file that contains information about the image data as specified in the previous section

`fusion_method`: there are 5 choices available and you must use the exact same naming/spelling as them: concatenation, tensor, cosine_similarity, dcca, cross_attention

You can change the hyperparameters to be tuned under the `Tuned parameters` section in `main.py`, by adding more values for each hyperparameter. You can also change things like "number of epochs" under the `Not tuned parameters` section. 

If you are not using NLST's CT image data, you may want to change `roi_size` and `resampled_spacing` to best reflect your data.

## When modality 1 is image data and modality 2 is also image data

### Image data input csv

The csv file should contain the following columns:

`pid`: patient/case ID

`m1_image_path`: path to modality 1's image scan in `.nii.gz` format

`m2_image_path`: path to modality 2's image scan in `.nii.gz` format

`nodule_path`: path to the nodule contour annotation in `.nii.gz` format

`pixel_X`: coordinate X of the nodule's centroid, in pixel (instead of in real-world coordinates)

`pixel_Y`: coordinate Y of the nodule's centroid, in pixel (instead of in real-world coordinates)

`pixel_Z`: coordinate Z of the nodule's centroid, in pixel (instead of in real-world coordinates)

`diagnosis_nodule`: nodule malignancy label, with 1 being maglignant, 0 being non-malignant 

The word `workspace` that you see in the sample/toy csv file provided by Rina means `/radraid/ruiwending`.

### Run model
Both images use 3D ViT as the encoders. However, if you want to use a simple CNN (the encoder from VQ-VAE), then please refer to Rina's uncleaned code at `/radraid/ruiwending/modeling/prostate_data_fusion/ucla_pirads/cnn_version/model.py`

```
cd m1_images_m2_images
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --path_to_image_data_csv ../prostate_data.csv --fusion_method <fusion_method>
```

`path_to_image_data_csv`: path to the csv file that contains information about the image data as specified in the previous section

`fusion_method`: there are 5 choices available and you must use the exact same naming/spelling as them: concatenation, tensor, cosine_similarity, dcca, cross_attention

You can change the hyperparameters to be tuned under the `Tuned parameters` section in `main.py`, by adding more values for each hyperparameter. You can also change things like "number of epochs" under the `Not tuned parameters` section. 

If you are not using UCLA IDx's MRI data, you may want to change `roi_size` and `resampled_spacing` to best reflect your data.

## UCLA IDx prostate MRI data info (just in case you need it)

- Clinial data

```
/radraid/ruiwending/prostate_data/updated_feb_2025
```

`clinical.csv`: contains all the clinical variables except for PSA values.

`IDx_Prostate_PSA_FULL_20250214.csv`: PSA values.

`Data Dictionary.pdf` and `User Document.pdf`: the documentation provided by the IDx 

You don't really need to use the other files under this folder.

- MRI data and nodule contour anontations

`/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations` contains the first batch and `/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations_feb_2025` contains the second batch. These 2 batches do not overlap and they are just cases that were collected at an older or more recent time. These are the cases that have nodule contour annotations and MRI data (T2W and ADC sequences) available.

Within each case, the only things you need are:

`image_n4_corrected.nii.gz`: N4 bias field corrected T2W sequence

`registered*adc.nii`: * means any strings but most importantly, this file must start with "registered". This file is ADC sequences registered to T2W, and the reason for the registration is that the nodule contour annotations were done on T2W, and we want to apply the annotations to ADC as well. The registration was done in Matlab using affine registration, and the code is under `/radraid/ruiwending/prostate_data/register_adc_to_t2w`

`Lesion*.nii.gz`: the nodule contour annotation for the cases under `/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations`

`Prostate.nii.gz`: the prostate contour annotation for the cases under `/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations`

`TgLs*.nii.gz`: the nodule contour annotation for the cases under `/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations_feb_2025`

`Prost*.nii.gz`: the prostate contour annotation for the cases under `/radraid/ruiwending/prostate_data/cleaned_cases_with_annotations_feb_2025`

## PICAI data

This is a publicly available prostate MRI cohort from [here](https://github.com/DIAGNijmegen/picai_labels)

Rina already downloaded the images and annotations and put them under `/radraid/ruiwending/prostate_data/prostate_picai_data`

Specifically, the images are under `./dataset0`, `./dataset1`, `./dataset2`, `./dataset3`, `./dataset4`

The nodule contour annotations are under `./csPCa_lesion_delineations`

The prostate contour annotations are under `./anatomical_delineations/whole_gland`

The clinical features are under `./clinical_information`

