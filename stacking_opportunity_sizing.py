import torch
import src.datasets as datasets
import json
import argparse
import os

def sizeOpportunity(model1,model2,dataset,args):
    preprocess_fn = model1.val_preprocess
    data_location = args.data_location
    data_loader = getDataloader(dataset,preprocess_fn,data_location)
    results = {}
    num_correct = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in data_loader:
        image,label = batch['images'],batch['labels']
        for item in [image,label,model1,model2]:
            item = item.to(args.device)
        alpha,loss,correct = getAndUseOptimalAlpha(image,label,loss_fn,model1,model2,args)
        num_correct += (1 if correct == True else 0)
        total += label.size(0)
    # print(alpha,loss,correct)  # Diagnostic - are these reasonable?
    results['optimalAlpha'] = num_correct / total
    results['model1'] = get_top1_acc(model1,dataset,args.batch_size,args)
    results['model2'] = get_top1_acc(model2,dataset,args.batch_size,args)

    print(results)
    with open(args.results_db,'a+') as f:
        f.write(json.dumps(results))
    return results

def getDataloader(dataset,preprocess_fn,data_location,batch_size=1,is_train=False,**kwargs):
    dataset_class = getattr(datasets,dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=data_location,
        batch_size=batch_size
    )
    data_loader = dataset.train_loader if is_train else dataset.test_loader
    return data_loader

def getAndUseOptimalAlpha(image,label,loss_fn,model1,model2,args):
    logits1,logits2 = model1(image),model2(image)
    optimalAlpha = getOptimalAlpha(logits1,logits2,label,loss_fn)
    ensembled_logits = optimalAlpha*logits1 + optimalAlpha*logits2
    ensemble_loss = loss_fn(ensembled_logits,label)
    pred = ensembled_logits.argmax(dim=1).to(args.device)
    correct = pred.eq(label.view_as(pred)).sum().item()  # Likely to cause problems I think
    total = label.size(0)
    correct = (correct == total)  # Convert to boolean
    return optimalAlpha, ensemble_loss, correct

def getOptimalAlpha(logits1,logits2,label,loss_fn):
    pass

def get_top1_acc(model,dataset,batch_size,args):
    data_loader = getDataloader(dataset,model.val_preprocess,args.data_location,batch_size)
    num_correct,total = 0,0
    for batch in data_loader:
        images,labels = batch['images'],batch['labels']
        for item in [images,labels,model]:
            item = item.to(args.device)
        logits = model(images)
        pred = logits.argmax(dim=1).to(args.device)
        num_correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)
    top_1_acc = num_correct / total
    return top_1_acc

def parse_arguments3():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('./data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='ImageNet',
    )
    parser.add_argument(
        "--model_ckpts",
        default=None,
        type=lambda x: x.split(","),
        help='Ckpt files for models being ensembled. Split by comma, e.g. "$pathStart$model1","$pathStart$model2" if pathStart=./models/wiseft/ViTB32_20/, model1=checkpoint_1.pt, model2=checkpoint_10.pt',
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args

if __name__ == "__main__":
    args = parse_arguments3()
    model1,model2,dataset = torch.load(args.model_ckpts[0]),torch.load(args.model_ckpts[1]),args.dataset
    sizeOpportunity(model1,model2,dataset,args)