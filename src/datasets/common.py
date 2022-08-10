import os
import torch
import json
import glob
import collections
import random
import time
import pickle

import numpy as np

from tqdm import tqdm

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes-1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].cuda())
            # print(features.shape) # Want 512-D for features, want number of classes (ImageNet ==> 1000) as dimension if producing logits

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device, model_name=None):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if image_encoder.__class__.__name__ is 'ImageClassifier':
        image_encoder2 = image_encoder.image_encoder   # Used when getting logits using models instead of features using image encoders
    else:
        image_encoder2 = image_encoder
    if image_encoder2.cache_dir is not None:
        if model_name is None:
            cache_dir = f'{image_encoder2.cache_dir}/{dname}/{split}'
        else:
            cache_dir = f'{image_encoder2.cache_dir}/{dname}/{split}/{model_name}' # Different models don't load from each other's logits
        cached_files = glob.glob(f'{cache_dir}/*')
    if image_encoder2.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file) # torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        # with open(f'test.pt','wb') as file:
        #     pickle.dump([0.1]*int(1.3e9), file, protocol=4) # Allows for larger dumps
        # print("Large data-writing test complete.")
        data = get_features_helper(image_encoder, loader, device)  # This is image_encoder, NOT image_encoder2. If a full
        # model is passed in instead of just an image encoder, this saves logits, not features.
        if image_encoder2.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                if name=='features' and model_name is not None:   # model_name is not None <==> this is for logit ensembling
                    torch.save(val, f'{cache_dir}/logits.pt',pickle_protocol=4)
                else:
                    torch.save(val, f'{cache_dir}/{name}.pt',pickle_protocol=4) # Allows for larger dumps
                # with open(f'{cache_dir}/{name}.pt','wb') as file:
                #     pickle.dump(val, file, protocol=4) # Allows for larger dumps
                    
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device, model_name=None):
        self.data = get_features(is_train, image_encoder, dataset, device, model_name)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader