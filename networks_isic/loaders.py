import torch
import torchvision
import myutils

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def get_loaders(dfs, size=100, batch_size=1, num_workers=1):
    """
    Function that takes a dictionary of dataframes and
    returns 2 dictionaries of pytorch dataloaders and dataset_sizes
    """
    # Reproducibility
    myutils.myseed(seed=42)

    # Custom pytorch dataloader for this dataset
    class Derm(Dataset):
        """
        Read a pandas dataframe with
        images paths and labels
        """
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            try:
                # Load image data and get label
                X = Image.open(self.df['image_path'][index]).convert('RGB')
                y = torch.tensor(int(self.df['label_code'][index]))
                print(f'{self.df['image_path'][index]}\t{self.df['label_code'][index]}')
            except IOError as err:
                pass

            if self.transform:
                X = self.transform(X)

            return X, y

    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transforms
    data_transforms = {'train' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)]),
                       'val' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)])}

    # Sets
    image_datasets = {x: Derm(dfs[x], transform=data_transforms[x]) for x in dfs.keys()}
    # Sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in dfs.keys()}
    # Loaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size, num_workers) for x in dfs.keys()}

    return dataloaders, dataset_sizes
