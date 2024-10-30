import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
import cv2

class ACDCDataset(Dataset):
    def __init__(self, data_dir, size=224, num_frame=8, channel=0):
        self.data_dir = data_dir
        self.size = size
        self.channel = channel
        self.num_frame = num_frame
        
        self.data_path = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        self._load_data_path()

    def _load_data_path(self):
        patients = os.listdir(self.data_dir)
        for patient in patients:
            patient_dir = os.path.join(self.data_dir, patient)
            if os.path.isdir(patient_dir):
                video_file = os.path.join(patient_dir, f"{patient}_4d.nii.gz")
                if os.path.isfile(video_file):
                    self.data_path.append(video_file)
                    label = self._extract_label(patient_dir)
                    self.labels.append(label)

        self.labels = self.label_encoder.fit_transform(self.labels)
    
    def mri_transform(self, data,):
        normalized_video = []
        step_size = data.shape[3] // self.num_frame
        data = data[:, :, :, np.round(np.arange(0, data.shape[3], step_size)).astype(int)[:self.num_frame]]
        
        for i in range(data.shape[3]):
            resized_image_slice = resize(data[:, :, self.channel, i], (self.size, self.size), order=0, preserve_range=True) 
            min_val = np.min(resized_image_slice)
            max_val = np.max(resized_image_slice)
            normalized_video.append(2 * (resized_image_slice - min_val) / (max_val - min_val) - 1)
        
        normalized_video = np.stack(normalized_video, axis=2)
        return normalized_video

    def _extract_label(self, patient_dir):
        info_file = os.path.join(patient_dir, 'Info.cfg')
        label = None
        with open(info_file, 'r') as f:
            for line in f:
                if line.startswith('Group:'):
                    label = line.split(':')[1].strip()
                    break
        return label

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        video_4d = nib.load(self.data_path[idx]).get_fdata()
        label = self.labels[idx]

        video = self.mri_transform(video_4d)

        # Convert to tensor
        video = torch.tensor(video, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return video, label
    
    def _save_samples(self, idx):
        data = nib.load(self.data_path[idx]).get_fdata()
        step_size = data.shape[3] // self.num_frame
        data = data[:, :, 0, np.round(np.arange(0, data.shape[3], step_size)).astype(int)[:self.num_frame]]
        data = (255 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(np.uint8)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter('output_video_{}.mp4'.format(idx), fourcc, 1, (data.shape[1], data.shape[0]))  # 1 fps, frame size (width, height)

        # Convert each frame from grayscale to RGB and write to video
        for i in range(data.shape[2]):
            frame = data[:, :, i]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            out.write(frame_rgb)

        # Release the VideoWriter object
        out.release()
    

if __name__ == "__main__":
    data_dir = '/home/comp/zhenshun/datasets/ACDC/database/training/'
    dataset = ACDCDataset(data_dir, 224, 8, 0)
    video, img = dataset[5]
    dataset._save_samples(0)
    dataset._save_samples(40)
    dataset._save_samples(80)