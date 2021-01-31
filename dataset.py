import os
import numpy as np
import cv2
import torch
import ast
import copy

from torch.utils.data import Dataset
from utils import reshape_img, NeedleAugmentation


COLOR_MAP = {
    "ETT - Abnormal": (255, 0, 0),
    "ETT - Borderline": (0, 255, 0),
    "ETT - Normal": (0, 0, 255),
    "NGT - Abnormal": (255, 255, 0),
    "NGT - Borderline": (255, 0, 255),
    "NGT - Incompletely Imaged": (0, 255, 255),
    "NGT - Normal": (128, 0, 0),
    "CVC - Abnormal": (0, 128, 0),
    "CVC - Borderline": (0, 0, 128),
    "CVC - Normal": (128, 128, 0),
    "Swan Ganz Catheter Present": (128, 0, 128),
}


class TrainDataset(Dataset):
    def __init__(self, root, df, transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.img_idx = df["StudyInstanceUID"].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = transform
        self.train_path = os.path.join(self.root, "train")
        # self.needle_path = os.path.join(self.root, "needle_aug")
        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        file_path = f"{self.train_path}/{file_name}.jpg"
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        # image = reshape_img(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # image = NeedleAugmentation(image, n_needles=2, dark_needles=False, p=0.3, needle_folder=self.needle_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]
        return image, label


class ValidDataset(Dataset):
    def __init__(self, root, df, transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.img_idx = df["StudyInstanceUID"].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = transform
        self.train_path = os.path.join(self.root, "train")
        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        file_path = f"{self.train_path}/{file_name}.jpg"
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        # image = reshape_img(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]
        return image, label


class SegDataset(Dataset):
    def __init__(self, root, df, flip_transform=None, target_cols=None):
        self.root = root
        self.img_idx = df["StudyInstanceUID"].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = flip_transform
        # self.norm_transform = norm_transform
        # self.img_path = os.path.join(self.root, "train")
        self.img_path = os.path.join(self.root, "train")
        self.lung_mask_path = os.path.join(self.root, "train_lung_masks")
        self.tube_mask_path = os.path.join(self.root, "train_tube_masks")

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        file_name = self.img_idx[idx]
        img_path = f"{self.img_path}/{file_name}.jpg"
        lung_mask_path = os.path.join(self.lung_mask_path, f"{file_name}.jpg")
        tube_mask_path = os.path.join(self.tube_mask_path, f"{file_name}.jpg")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # image = np.expand_dims(image, axis=2)
        lung_mask = cv2.imread(lung_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = lung_mask > 0
        lung_mask[mask] -= 100
        # lung_mask = np.expand_dims(lung_mask, axis=2)

        tube_mask = cv2.imread(tube_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = tube_mask > 100
        tube_mask[mask] -= 50
        # tube_mask = np.expand_dims(tube_mask, axis=2)

        # img = np.concatenate([image, tube_mask, lung_mask], axis=2)
        img[:, :, 1] += lung_mask
        img[:, :, 0] += tube_mask

        if self.transform:
            # augmented = self.transform(image=image, masks=[tube_mask, lung_mask])
            augmented = self.transform(image=img)
            image = augmented["image"]
            # img_masks = augmented["masks"]

        label = self.labels[idx]
        return image, label


class AnnotDataset(Dataset):
    def __init__(self, root, df, df_annotations, flip_transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.df_annotations = df_annotations
        self.file_names = df["StudyInstanceUID"].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = flip_transform
        self.img_path = os.path.join(self.root, "train")
        self.lung_mask_path = os.path.join(self.root, "train_lung_masks")

        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.file_names[idx]
        file_path = f"{self.img_path}/{uid}.jpg"
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        mask = np.zeros_like(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # masks = [np.zeros_like(image) for _ in range(2)]
        # masks[1] = cv2.imread(f"{self.lung_mask_path}/{uid}.jpg", cv2.IMREAD_GRAYSCALE)
        has_mask = False
        query_string = f"StudyInstanceUID == '{uid}'"
        df = self.df_annotations.query(query_string)

        if df.shape[0] > 0:
            has_mask = True
            for _, row in df.iterrows():
                # label = self.label_map[row["label"]]
                data = ast.literal_eval(row["data"])
                ctr_cord = np.array([[np.array(x) for x in data]])
                mask = cv2.polylines(mask, ctr_cord, isClosed=False, color=(255,), thickness=10)
                image = cv2.polylines(image, ctr_cord, isClosed=False, color=COLOR_MAP[row["label"]], thickness=5)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            mask = augmented["mask"]
            image = augmented["image"]
        mask = mask.unsqueeze(0).float() / 255.0
        # for ii, amask in enumerate(tmasks):
        #     masks[ii] = torch.from_numpy(amask).unsqueeze(0).float() / 255.0
        # mask = torch.from_numpy(np.concatenate(masks, axis=0)).float()

        return image, mask, self.labels[idx], has_mask


class AnnotDatasetS2(Dataset):
    def __init__(self, root, df, df_annotations, flip_transform=None, target_cols=None):
        self.root = root
        self.df = df
        self.df_annotations = df_annotations
        self.file_names = df["StudyInstanceUID"].values
        self.labels = torch.from_numpy(df[target_cols].values).float()
        self.transform = flip_transform
        self.img_path = os.path.join(self.root, "train")
        self.tube_mask_path = os.path.join(self.root, "train_tube_masks")

        self.clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.file_names[idx]
        img_path = os.path.join(f"{self.img_path}", f"{uid}.jpg")
        # tube_mask_path = os.path.join(self.tube_mask_path, f"{uid}.jpg")

        # read image and preprocess
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = self.clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        annot_image = image.copy()
        query_string = f"StudyInstanceUID == '{uid}'"
        df = self.df_annotations.query(query_string)

        if df.shape[0] > 0:
            for _, row in df.iterrows():
                data = ast.literal_eval(row["data"])
                ctr_cord = np.array([[np.array(x) for x in data]])
                # mask = cv2.polylines(mask, ctr_cord, isClosed=False, color=(255,), thickness=10)
                annot_image = cv2.polylines(annot_image, ctr_cord, isClosed=False, color=COLOR_MAP[row["label"]], thickness=5)

        # tube_mask = cv2.imread(tube_mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = tube_mask < 100
        # tube_mask[mask] = 0
        if self.transform:
            augmented = self.transform(image=image, image_annot=annot_image)
            # tube_mask = augmented["mask"]
            annot_image = augmented["image_annot"]
            image = augmented["image"]
        # tube_mask = tube_mask.unsqueeze(0).float() / 255.0

        return image, annot_image, self.labels[idx]
