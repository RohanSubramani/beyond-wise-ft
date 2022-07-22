from src.args2 import parse_arguments2
from src.models.modeling import * # Includes ImageClassifier, ResNet

def trainStackingClassifier(args):
    dataset = getDataset(args.original_dataset,args.model_ckpts)
    stackingClassifier = getStackingClassifier(args.model_class,args.model_details)
    train(stackingClassifier,dataset)


def getDataset(original_dataset,model_ckpts):
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    all_all_logits = []
    # all_logits = torch.tensor([model(input) for model in models])
    pass

def getStackingClassifier(model_class,model_details):
    pass

def train(stackingClassifier,dataset):
    pass

if __name__ == "__main__":
    args = parse_arguments2()
    trainStackingClassifier(args)