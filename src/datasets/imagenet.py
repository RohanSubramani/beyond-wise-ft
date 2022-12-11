import os
import torch

from .common import ImageFolderWithPaths, ImageFolderWithPaths2, SubsetSampler
from .imagenet_classnames import get_classnames
import numpy as np

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass
    
    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"ImageNet{k}"
    dyn_cls = type(cls_name, (ImageNetK, ), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls

class DeterministicImageNet(ImageNet):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.populate_train()
        self.populate_test()
    
    def populate_train(self): # bad hacky thing: pass in a subset function
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : False} if sampler is None else {}
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

class DeterministicImageNetWithLogits(DeterministicImageNet):
    def __init__(self,
                 preprocess,
                 all_logits,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai',
                 subset_proportion=1.0,
                 is_train=True):
        self.preprocess = preprocess
        self.all_logits = all_logits
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.subset_proportion = subset_proportion
        self.is_train=is_train

        if is_train:
            self.populate_train(subset_proportion, all_logits)
        else:
            self.populate_test(all_logits)
    
    def populate_train(self,subset_proportion,all_logits): # pass in images instead of constricting traindir from scratch
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths2(
            traindir,
            all_logits=all_logits,  # Only in this dataset. Always contains logits for entire dataset, subsetting happens below.
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {} # This should be fine, since the logits are already matched with the images
        if subset_proportion == 1.0:
            train_subset = self.train_dataset
            batch_size = self.batch_size
        else: # make this into a function to pass around
            assert subset_proportion < 1.0 and subset_proportion > 0.0, "Invalid subset proportion, should be between 0.0 and 1.0"
            sample = getEvenlySpacedSample(int(subset_proportion*len(self.train_dataset)),len(self.train_dataset))
            train_subset = torch.utils.data.Subset(self.train_dataset, sample)
            batch_size = min(len(sample),self.batch_size)

        self.train_loader = torch.utils.data.DataLoader(
            train_subset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )
    
    def populate_test(self,all_logits):
        # print("Populating test dataset of DeterministicImageNetWithLogits")
        self.test_dataset = self.get_test_dataset(all_logits)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()  # None
        )
        # print(f"DeterministicImageNetWithLogits: len(self.test_loader)={len(self.test_loader)}")

    def get_test_dataset(self,all_logits):
        return ImageFolderWithPaths2(self.get_test_path(), all_logits, transform=self.preprocess)

def getEvenlySpacedSample(sample_size,population_size):
    space = int(float(population_size)/float(sample_size))
    i = 0
    sample = []
    while i < population_size:
        sample.append(i)
        i += space
    return sample