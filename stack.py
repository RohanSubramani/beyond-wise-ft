from src.args2 import parse_arguments2
from src.models.modeling import * # Includes ImageClassifier, ResNet
import src.datasets as datasets
from open_clip.src.open_clip.transform import image_transform

def trainStackingClassifier(args):
    model, input_key, preprocess_fn, image_enc = getStackingClassifier(args.model_class,args.model_details)
    dataset = getDataset(args.original_dataset,args.model_ckpts,preprocess_fn)
    train(model,dataset,args)


def getStackingClassifier(model_class,model_details):
    if args.load is not None:
        image_classifier = ImageClassifier.load(args.load)  # args.load is here for the alpha model ckpt

        if args.freeze_encoder:
            print('Fine-tuning a linear classifier')
            model = image_classifier.classification_head
            input_key = 'features'
            preprocess_fn = image_classifier.val_preprocess # not train_preprocess, because data aug isn't needed if learned features are fixed
            image_enc = image_classifier.image_encoder
        else:
            print('Fine-tuning end-to-end')
            model = image_classifier
            input_key = 'images'
            preprocess_fn = image_classifier.train_preprocess
            image_enc = None
            image_classifier.process_images = True

        return model, input_key, preprocess_fn, image_enc


def getDataset(original_dataset,model_ckpts,preprocess_fn):
    dataset_class = getattr(datasets, original_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    # TODO Add get_dataloader here, etc.
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    all_all_logits = []
    # all_logits = torch.tensor([model(input) for model in models])

def train(stackingClassifier,dataset):
    pass

if __name__ == "__main__":
    args = parse_arguments2()
    trainStackingClassifier(args)