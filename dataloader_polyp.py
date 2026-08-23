import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from prime_selection import run_prime_selection


class PolypDataset(data.Dataset):
    """
    Unified adaptive dataloader for polyp segmentation with integrated PRIME selection.
    Uses Albumentations and strictly handles binary mask conversion.
    """
    def __init__(
        self,
        image_root,
        gt_root,
        trainsize,
        augmentation,
        split="train",
        color_image=True,
        enable_prime=False,
        cc_th=0.8,
        per=1,
        similarity_matrix_path=None
    ):
        self.trainsize = trainsize
        self.color_image = color_image
        self.augmentation = augmentation
        self.split = split

        # Load and sort file paths
        exts = (".jpg", ".png", ".jpeg", ".tif")
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(exts)])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(exts)])

        self.filter_files()

        # Apply PRIME Data Selection on training split
        if self.split == "train" and enable_prime:
            sim_mat = np.load(similarity_matrix_path) if similarity_matrix_path and os.path.exists(similarity_matrix_path) else None
            selected_ids, _ = run_prime_selection(
                self.images,
                cc_th=cc_th,
                per=per,
                im_size=(self.trainsize, self.trainsize),
                similarity_matrix=sim_mat
            )
            self.images = [self.images[i] for i in selected_ids]
            self.gts = [self.gts[i] for i in selected_ids]

        self.size = len(self.images)

        # Transformation Setup
        mean = [0.485, 0.456, 0.406] if color_image else [0.5]
        std = [0.229, 0.224, 0.225] if color_image else [0.229]

        if self.split == "train" and self.augmentation:
            self.transform = A.Compose([
                A.Rotate(limit=90, p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def filter_files(self):
        valid_images, valid_gts = [], []
        for img_p, gt_p in zip(self.images, self.gts):
            if os.path.exists(img_p) and os.path.exists(gt_p):
                valid_images.append(img_p)
                valid_gts.append(gt_p)
        self.images, self.gts = valid_images, valid_gts

    def __getitem__(self, index):
        # 1. Load Image
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY)

        # 2. Load Mask
        mask_np = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        # 3. Apply Transformations
        augmented = self.transform(image=image, mask=mask_np)
        image = augmented["image"]
        mask = augmented["mask"]

        # 4. Adaptive Binary Mask Logic
        max_val = mask.max()
        if max_val > 127.0:
            mask = (mask > 20).long()
        else:
            mask = (mask >= 1).long()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # 5. Return Logic
        if self.split == "train":
            return image, mask
        else:
            with Image.open(self.gts[index]) as img:
                original_shape = img.size
            name = os.path.basename(self.images[index])
            if name.lower().endswith(".jpg"):
                name = name.rsplit(".", 1)[0] + ".png"
            return image, mask, original_shape, name

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    augmentation=False,
    split="train",
    color_image=True,
    enable_prime=False,
    cc_th=0.8,
    per=1,
    similarity_matrix_path=None
):
    dataset = PolypDataset(
        image_root=image_root,
        gt_root=gt_root,
        trainsize=trainsize,
        augmentation=augmentation,
        split=split,
        color_image=color_image,
        enable_prime=enable_prime,
        cc_th=cc_th,
        per=per,
        similarity_matrix_path=similarity_matrix_path
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )