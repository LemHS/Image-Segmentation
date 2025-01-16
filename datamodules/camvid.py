import torch
import torch.nn.functional as F
import lightning as L
import os
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader, random_split
    
class CamVidDataset(Dataset):
    def __init__(self, img_dir, rgb2class):
        self.img_dir = img_dir
        self.mask_dir = f"{img_dir}_labels"
        self.rgb2class = rgb2class

        self.img_files = os.listdir(self.img_dir)
        self.mask_files = [img_file.replace(".png", "_L.png") for img_file in self.img_files]

        self.img_paths = [f"{self.img_dir}/{img_file}" for img_file in self.img_files]
        self.mask_paths = [f"{self.mask_dir}/{mask_file}" for mask_file in self.mask_files]

        img = cv2.imread(self.img_paths[0])
        max_row = img.shape[0] // 256 + 1
        max_col = img.shape[1] // 256 + 1
        self.n_patch = max_col * max_row

    def __len__(self):
        return len(self.img_paths) * self.n_patch

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset with length {len(self)}.")
        patch_index = index % self.n_patch
        index = index // self.n_patch

        img = cv2.imread(self.img_paths[index])[:,:,::-1].copy()
        img = self._get_patch(img, patch_index)
        img = torch.from_numpy(img).float().permute(-1, 0, 1)
        img = (img - 0)/(255 - 0)*(1 - 0) + 0

        mask = cv2.imread(self.mask_paths[index])[:,:,::-1].copy()
        mask = self._get_patch(mask, patch_index)
        for rgb, class_idx in self.rgb2class.items():
            change = (mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])
            mask[change] = class_idx
        mask = torch.from_numpy(mask).long().permute(-1, 0, 1)

        if (img.shape[1] != 256) or (img.shape[2] != 256):
            pad_bottom = 256 - img.shape[1]
            pad_right = 256 - img.shape[2]
            img = F.pad(img, (0, pad_right, 0, pad_bottom), mode="reflect")
            mask = F.pad(mask, (0, pad_right, 0, pad_bottom), mode="reflect")

        mask = mask[0, :, :]
        
        return img, mask
    
    def _get_patch(self, img, patch_index):
        max_col = img.shape[1] // 256 + 1
        row_idx = patch_index // max_col
        col_idx = patch_index % max_col
        
        patch = img[256 * row_idx:256 * (row_idx + 1), 256 * col_idx:256 * (col_idx + 1)]

        return patch
    
class CamVidDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_dir: str, 
                 val_dir: str, 
                 test_dir: str, 
                 class_dict: pd.DataFrame, 
                 train_batch_size: int = 4, 
                 val_test_batch_size: int = 512):
        
        super().__init__()
        

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.class_dict = class_dict
        self.n_classes = len(class_dict)
        self.rgb2class = {tuple(rgb): class_idx for class_idx, rgb in enumerate(class_dict[["r", "g", "b"]].values)}
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_set = CamVidDataset(self.train_dir, self.rgb2class)
        self.val_set = CamVidDataset(self.val_dir, self.rgb2class)
        self.test_set = CamVidDataset(self.test_dir, self.rgb2class)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_test_batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.val_test_batch_size, shuffle=False)
    
def create_camvidmodule():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "data", "CamVid", "train")
    val_dir = os.path.join(base_dir, "data", "CamVid", "val")
    test_dir = os.path.join(base_dir, "data", "CamVid", "test")
    class_dict_path = os.path.join(base_dir, "data", "CamVid", "class_dict.csv")
    class_dict = pd.read_csv(class_dict_path)

    return CamVidDataModule(train_dir, val_dir, test_dir, class_dict, val_test_batch_size=11)