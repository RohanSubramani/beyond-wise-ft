from PIL import Image

from imagenetv2_pytorch import ImageNetV2Dataset

from .imagenet import ImageNet

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
        super().__init__(*args,**kwargs)  
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

class ImageNetV2WithLogits(ImageNet):
    def get_test_dataset(self,all_logits):
        return ImageNetV2DatasetWithPaths2(all_logits,transform=self.preprocess, location=self.location)
