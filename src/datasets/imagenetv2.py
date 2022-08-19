from PIL import Image
import os

from imagenetv2_pytorch import ImageNetV2Dataset
from .imagenet import ImageNet, DeterministicImageNetWithLogits
from .imagenet_classnames import get_classnames


class ImageNetV2DatasetWithPaths(ImageNetV2Dataset):
    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'images': img,
            'labels': label,
            'image_paths': str(self.fnames[i])
        }

class ImageNetV2(ImageNet):
    def get_test_dataset(self):
        return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location)

class ImageNetV2DatasetWithPaths2(ImageNetV2DatasetWithPaths):  # For logit ensembling
    def __init__(self,all_logits,*args,**kwargs):
        super().__init__(*args,**kwargs)   # Default initialization of parent class (which itself inherits from at least one parent class)
        self.all_logits = all_logits

    def __getitem__(self, index):
        d = super(ImageNetV2DatasetWithPaths2, self).__getitem__(index)
        image, label, image_path = d['images'], d['labels'], d['image_paths']
        all_logits = self.all_logits[index]
        return {
            'images': image,
            'all_logits': all_logits,
            'labels': label,
            'image_paths': image_path
        }

class ImageNetV2WithLogits(DeterministicImageNetWithLogits):
    def __init__(self,
                 preprocess,
                 all_logits,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.all_logits = all_logits
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.populate_test(all_logits)   # See ImageNet populate test 
    
    def get_test_dataset(self,all_logits):
        return ImageNetV2DatasetWithPaths2(all_logits,transform=self.preprocess, location=self.location)