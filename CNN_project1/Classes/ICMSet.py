import scipy.io as sio
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os as os
from torchvision.datasets import VisionDataset


class ICMSet(VisionDataset):
    """Dataset Class created by Joanna Molad F.
    Class similar to ImageFolder of Pytorch but for MATLAB file type.
    Inputs:
    1) img_folder: Dataset path where subfolders are the different classes just like ImageForlder of Pytorch.
    2) transform: Optional input.

    Outputs the dataset, either as a list of the image paths and their labels, or a single sample.

    """
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,):

        super(ICMSet, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
#       self.targets = [s[1] for s in samples]

    def find_classes(self, directory):
        try:
            classes_list = os.listdir(directory)
        except:
            print("Either the directory entered does not exist or there are no subfolders.")

        classes_dict = {cls_name: i for i, cls_name in enumerate(classes_list)}


        return classes_list, classes_dict

    def make_dataset(self, directory, classes_dict):
        """Returns list of path to files in dataset and their labels
            data_list = (filepath, label)"""
        data_list = []
        #available_classes = set()
        for target_class in sorted(classes_dict.keys()):
            class_index = classes_dict[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    data_list.append(item)

        return data_list

# The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        matfile = sio.loadmat(path)
        # sample = matfile.get('strial')
        sample = matfile.get('singleTrial')
        # sample = matfile.get('all_matrices')
        sample_size = sample.shape
        sample_size = list(sample_size)
        depth = sample_size[2]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label
