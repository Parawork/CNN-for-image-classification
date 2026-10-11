"""
Everything to do with the data.

We went with EuroSAT (RGB): 27,000 Sentinel-2 satellite patches across 10
land-use classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake). It suits the "edge sensor"
story well - think of a small satellite or drone that has to classify what it
sees on board instead of beaming every image down.

  * Images are kept at 64 x 64 x 3, the largest size the assignment allows.
  * Split is stratified 70 / 15 / 15 with a fixed seed. The indices are written
    to data/splits.json the first time, so every model - ours and the
    pre-trained ones - sees exactly the same train / val / test images.
"""
import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from .paths import DATA_DIR

DATA_ROOT = DATA_DIR
CACHE_FILE = DATA_DIR / "eurosat64.pt"
SPLIT_FILE = DATA_DIR / "splits.json"
IMG_SIZE = 64
SEED = 42

# ImageNet statistics: needed by the pre-trained networks, and harmless for the
# custom networks, so one normalisation is used for all models.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_eurosat(root=DATA_ROOT):
    """Download EuroSAT once, downscale to 64x64 and cache it as a uint8 tensor.

    Returns (images[N,3,64,64] uint8, labels[N] int64, class_names).
    """
    if CACHE_FILE.exists():
        cache = torch.load(CACHE_FILE)
        return cache["images"], cache["labels"], cache["classes"]

    print("First run: downloading EuroSAT (~90 MB) and caching it at 64x64 - this happens only once.")
    ds = datasets.EuroSAT(root=str(root), download=True)
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

    images = torch.empty((len(ds), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    labels = torch.empty(len(ds), dtype=torch.long)
    for i in range(len(ds)):
        img, y = ds[i]
        images[i] = TF.pil_to_tensor(resize(img.convert("RGB")))
        labels[i] = y

    torch.save({"images": images, "labels": labels, "classes": ds.classes}, CACHE_FILE)
    return images, labels, ds.classes


def make_splits(labels, seed=SEED):
    """Stratified 70/15/15 split. Created once, then re-loaded from disk."""
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as f:
            return json.load(f)

    idx = list(range(len(labels)))
    y = labels.tolist()
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    temp_y = [y[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_y, random_state=seed)

    splits = {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f)
    return splits


class EuroSATTensorDataset(Dataset):
    """In-memory dataset (the whole 64x64 dataset is only ~330 MB as uint8).

    augment    : random horizontal / vertical flips and 90-degree rotations.
                 Satellite images have no canonical orientation, so these are
                 label-preserving and cost nothing to compute.
    input_size : the custom CNNs use 64. Pre-trained networks can optionally be
                 fed an upsampled copy (e.g. 128) - the information content is
                 still that of the 64x64 sensor image.
    """

    def __init__(self, images, labels, augment=False, input_size=IMG_SIZE):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.input_size = input_size
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255.0
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, k=int(torch.randint(0, 4, (1,))), dims=[1, 2])
        if self.input_size != IMG_SIZE:
            x = TF.resize(x, [self.input_size, self.input_size], antialias=True)
        return self.normalize(x), self.labels[i]


def get_dataloaders(batch_size=64, input_size=IMG_SIZE, augment=True, num_workers=0):
    """Return ({'train','val','test'} -> DataLoader, class_names)."""
    images, labels, classes = load_eurosat()
    splits = make_splits(labels)

    loaders = {}
    for name, idx in splits.items():
        idx = torch.tensor(idx)
        ds = EuroSATTensorDataset(images[idx], labels[idx],
                                  augment=augment and name == "train",
                                  input_size=input_size)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                                   num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loaders, classes
