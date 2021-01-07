import os
import numpy as np
import cv2
import torch
import ast

from torch.utils.data import Dataset
from utils import reshape_img, NeedleAugmentation


class TrainDataset(Dataset):
    def __init__(self, root, df, transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.img_idx = df['StudyInstanceUID'].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = transform
        self.train_path = os.path.join(self.root, "train")
        # self.needle_path = os.path.join(self.root, "needle_aug")
        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        file_path = f'{self.train_path}/{file_name}.jpg'
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        image = reshape_img(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # image = NeedleAugmentation(image, n_needles=2, dark_needles=False, p=0.3, needle_folder=self.needle_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        return image, label


class ValidDataset(Dataset):
    def __init__(self, root, df, transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.img_idx = df['StudyInstanceUID'].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = transform
        self.train_path = os.path.join(self.root, "train")
        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        file_path = f'{self.train_path}/{file_name}.jpg'
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        image = reshape_img(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        return image, label


class SegDataset(Dataset):
    def __init__(self, root, df, flip_transform=None, target_cols=None):
        self.root = root
        self.img_idx = df['StudyInstanceUID'].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = flip_transform
        # self.norm_transform = norm_transform
        # self.img_path = os.path.join(self.root, "train")
        self.img_path = os.path.join(self.root, "train")
        self.mask_path = os.path.join(self.root, "train_lung_masks")

        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        img_path = f'{self.img_path}/{file_name}.jpg'
        mask_path = f'{self.mask_path}/{file_name}.jpg'

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        image = self.reshape_img(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        img_mask = cv2.imread(mask_path, -1)
        img_mask = self.reshape_img(img_mask)
        img_mask = img_mask.astype(np.float32)/255.
        # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image, mask=img_mask)
            image = augmented["image"]
            img_mask = augmented["mask"]

        label = self.labels[idx]
        return image, img_mask, label

    def reshape_img(self, cl_img):
        ht, wd = cl_img.shape
        ww = max(ht+2,wd+2)
        hh = ww
        constant = np.zeros((hh,ww), dtype=np.uint8)
        xx = (ww-wd)//2
        yy = (hh-ht)//2
        constant[yy:yy+ht, xx:xx+wd] = cl_img

        target_area = 1024*1024
        ratio = float(constant.shape[1])/float(constant.shape[0])
        new_h = int(np.sqrt(target_area / ratio) + 0.5)
        new_w = int((new_h * ratio) + 0.5)

        res_img = cv2.resize(constant, (new_w,new_h))
        return res_img


class AnnotDataset(Dataset):
    def __init__(self, root, df, df_annotations, flip_transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.df_annotations = df_annotations
        self.file_names = df['StudyInstanceUID'].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = flip_transform
        self.img_path = os.path.join(self.root, "train")
        self.lung_mask_path = os.path.join(self.root, "train_lung_masks")

        # self.label_map = {}
        # for idx, lab_name in enumerate(target_cols):
        #     self.label_map[lab_name] = idx
        # self.label_map["lung"] = idx+1
        # self.label_map['background'] = idx+2
        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.file_names[idx]
        file_path = f'{self.img_path}/{uid}.jpg'
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        # TODO: yet to reshape image
        masks = [np.zeros_like(image) for _ in range(2)]
        masks[1] = cv2.imread(f"{self.lung_mask_path}/{uid}.jpg", cv2.IMREAD_GRAYSCALE)
        query_string = f"StudyInstanceUID == '{uid}'"
        df = self.df_annotations.query(query_string)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.astype(np.float32)

        for _, row in df.iterrows():
            # label = self.label_map[row["label"]]
            data = ast.literal_eval(row["data"])
            masks[0] = cv2.polylines(masks[0], np.array([[np.array(x) for x in data]]), isClosed=False, color=(255,), thickness=10)

        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            tmasks = augmented["masks"]
            image = augmented["image"]
            for ii, amask in enumerate(tmasks):
                masks[ii] = torch.from_numpy(amask).unsqueeze(0).float()/255.
            # masks = torch.from_numpy(np.concatenate(masks,axis=0)).float()

        return image, masks, self.labels[idx]
