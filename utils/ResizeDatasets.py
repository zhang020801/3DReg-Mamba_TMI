import os
import glob
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader

import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class MedicalImageDataset(Dataset):
    def __init__(self, files, target_size=(160, 192, 160), device='cpu'):
        
        self.files = files
        self.target_size = target_size
        self.device = device

    def __len__(self):
        return len(self.files)

    def resample_image(self, image):
        
        original_size = np.array(image.GetSize(), dtype=np.int32) 
        original_spacing = np.array(image.GetSpacing(), dtype=np.float32) 

        new_size = np.array(self.target_size, dtype=np.int32)  
        new_spacing = (original_size * original_spacing) / new_size  

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in new_size]) 
        resampler.SetOutputSpacing(tuple(new_spacing))  
        resampler.SetOutputOrigin(image.GetOrigin())  
        resampler.SetOutputDirection(image.GetDirection())  
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        resampled_image = resampler.Execute(image)
        return sitk.GetArrayFromImage(resampled_image)  

    def __getitem__(self, idx):
        
        file_path = self.files[idx]
        image = sitk.ReadImage(file_path)  
        image_array = self.resample_image(image)  
  
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        return image_tensor, os.path.basename(file_path)


class IXIImageDataset(Dataset):
    def __init__(self, files, target_size=(160, 192, 160), device='cpu'):
        
        self.files = files
        self.target_size = target_size
        self.device = device

    def __len__(self):
        return len(self.files)

    def resample_image(self, image):
        
        image = sitk.GetImageFromArray(image)

        original_size = np.array(image.GetSize(), dtype=np.int32)  
        original_spacing = np.array(image.GetSpacing(), dtype=np.float32)  

        new_size = np.array(self.target_size, dtype=np.int32)  
        new_spacing = (original_size * original_spacing) / new_size 

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in new_size])  
        resampler.SetOutputSpacing(tuple(new_spacing)) 
        resampler.SetOutputOrigin(image.GetOrigin())  
        resampler.SetOutputDirection(image.GetDirection()) 
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        resampled_image = resampler.Execute(image)
        return sitk.GetArrayFromImage(resampled_image)  

    def __getitem__(self, idx):
        
        file_path = self.files[idx]
        image,_ = pkload(file_path) 
        image_array = self.resample_image(image)  

        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        return image_tensor, os.path.basename(file_path)