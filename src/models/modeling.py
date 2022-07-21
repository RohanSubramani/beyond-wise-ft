import torch
import copy

# import clip.clip as clip
import open_clip.src.open_clip as open_clip

from src.models import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        try:
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained='laion400m_e32',data_augmentation_number=args.data_augmentation)
            print("Using pretrained=laion400m_e32")
        except RuntimeError:
            try:
                self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model,pretrained='laion400m_e31',data_augmentation_number=args.data_augmentation)
                print("Using pretrained=laion400m_e31")
            except RuntimeError:
                print("Pretrained model not found!")
                cont = input("Would you like to continue anyway? 1 = yes, 2 = no\n")
                if int(cont) == 2:
                    print("Stopping run.")
                    raise RuntimeError("No pretrained model found.")
                self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                    args.model,data_augmentation_number=args.data_augmentation)  # No pretrained sometimes? Hard to train from scratch though.
        self.model.to(args.device)
                                                                 # clip.load(args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
