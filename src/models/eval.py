import os
import json
import sys
import tqdm

import torch
import numpy as np

from src.models import utils
from src.models.modeling import *
from src.models.utils import saveAlphas
from src.datasets.common import get_dataloader, maybe_dictionarize
from stacking_opportunity_sizing import useLogitEnsemble, getOptimalAlpha, getDataloader as get_dataloader_for_ose_or_oae
import src.datasets as datasets
from inputimeout import inputimeout, TimeoutOccurred

def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
        # from datasets/common.py, if image_enc is not None:
        # feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        # dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None   
        # from datasets/common.py, if image_enc is None:
        # dataloader = dataset.train_loader if is_train else dataset.test_loader

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []
    
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            sys.stdout.write(f"\r{i+1}/{len(dataloader)}") # I do want this to be erased by whatever comes next
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            logits = utils.get_logits(x, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)
            
            loss = loss_fn(logits, y)
            val_loss += loss.item()

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    metrics['val_loss'] = val_loss
    
    return metrics

def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    print("Starting evaluation on eval datasets.")
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        # print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )

        results = eval_single_dataset(image_classifier, dataset, args)

        if 'top1' in results:
            sys.stdout.write(f"\t{dataset_name} Top-1 accuracy: {results['top1']:.4f}\n")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                sys.stdout.write(f"\t{dataset_name} {key}: {val:.4f}\n")
            info[dataset_name + ':' + key] = val
    if args.results_db is not None:
        if os.path.isfile(args.results_db):
            try:
                option = int(inputimeout(f"The entered results_db ({args.results_db}) already exists. Type '1' to add to it, and type '2' to overwrite it.",timeout=30))
            except TimeoutOccurred:
                option = 2
                
            if option == 1:
                with open(args.results_db, 'r') as f:
                    results = json.loads(f.read())
                results.append(info)
                with open(args.results_db, 'w') as f:
                    f.write(json.dumps(results))
            else:
                with open(args.results_db, 'w') as f:
                    f.write(json.dumps([info])) # List with info as its only element
        else:
            dirname = os.path.dirname(args.results_db)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps([info])) # List with info as its only element
            # print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def evaluate2(alphaModel, model_ckpts, args, final_model=False):  # For evaluation when stacking
    from stack import getLogitDataloader
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    preprocess_fn = models[0].val_preprocess
    if args.eval_datasets is None:
        return
    print("Starting evaluation on eval datasets.")
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        # print('Evaluating on', dataset_name)
        dataloader = getLogitDataloader(dataset_name,model_ckpts,preprocess_fn,args,is_train=False)

        results = eval_single_dataset2(alphaModel, dataloader, args, final_model, dataset_name)

        if 'top1' in results:
            sys.stdout.write(f"\t{dataset_name} Top-1 accuracy: {results['top1']:.4f}\n")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                sys.stdout.write(f"\t{dataset_name} {key}: {val:.4f}\n")
            info[dataset_name + ':' + key] = val
    if args.results_db is not None:
        if os.path.isfile(args.results_db):
            with open(args.results_db, 'r') as f:
                results = json.loads(f.read())
            results.append(info)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps(results))
        else:
            dirname = os.path.dirname(args.results_db)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps([info])) # List with info as its only element
            # print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def eval_single_dataset2(alphaModel, dataloader, args, final_model=False, dataset_name=None):  # For evaluation when stacking
    from stack import maybe_dictionarize2
    # print(f"alphaModel.__class__.__name__={alphaModel.__class__.__name__}") # ImageClassifier
    if alphaModel.__class__.__name__ is 'ImageClassifier' or 'ImageClassifier2':
        # alphaModel = ImageClassifier2(alphaModel.image_encoder, alphaModel.classification_head)
        if args.freeze_encoder:
            model = alphaModel.classification_head if hasattr(alphaModel,'classification_head') else alphaModel.module.classification_head
            # input_key = 'features'
            image_enc = alphaModel.image_encoder if hasattr(alphaModel,'image_encoder') else alphaModel.module.image_encoder
        else:
            model = alphaModel
            # input_key = 'images'
            image_enc = torch.nn.Identity()
        
        
        model.eval()
        batched_data = enumerate(dataloader)
        # print(f"len(dataloader)={len(dataloader)}")
        
        device = args.device
        image_enc.to(device)

        # if hasattr(dataset, 'post_loop_metrics'):
        #     # keep track of labels, predictions and metadata
        #     all_labels, all_preds, all_metadata = [], [], []
        
        loss_fn = torch.nn.CrossEntropyLoss()
        val_loss = 0
        all_val_alphas = []
        
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in batched_data:
                sys.stdout.write(f"\r{i+1}/{len(dataloader)}") # I do want this to be erased by whatever comes next
                data = maybe_dictionarize2(data)  # Keys in dict: images, all_logits, labels
                # x = [data[key].cuda() for key in list(data.keys())[:-1]]
                images = data['images'].to(device)
                encoded_images = image_enc(images)
                all_logits = data['all_logits'].to(device)
                y = data['labels'].to(device)

                # if 'image_paths' in data:
                #     image_paths = data['image_paths']
                # print(f"model.__class__.__name__={model.__class__.__name__}")  #  "ImageClassifier"

                # logits = model(*x) # utils.get_logits2(model,*x)  # 
                alphas = utils.get_alphas(model, encoded_images)
                all_val_alphas += alphas.tolist()
                weird_ensembled_logits = alphas @ all_logits
                fixed_ensembled_logits = torch.transpose(torch.diagonal(weird_ensembled_logits),0,1)
                
                # projection_fn = getattr(dataset, 'project_logits', None)
                # if projection_fn is not None:
                #     logits = projection_fn(logits, device)

                # if hasattr(dataset, 'project_labels'):
                #     y = dataset.project_labels(y, device)

                pred = fixed_ensembled_logits.argmax(dim=1).to(device)   #   , keepdim=True
                
                # if hasattr(dataset, 'accuracy'):
                #     acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                #     correct += acc1
                #     n += num_total
                # else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

                loss = loss_fn(fixed_ensembled_logits, y)
                val_loss += loss.item()

                # if hasattr(dataset, 'post_loop_metrics'):
                #     all_labels.append(y.cpu().clone().detach())
                #     all_preds.append(logits.cpu().clone().detach())
                #     metadata = data['metadata'] if 'metadata' in data else image_paths
                #     all_metadata.extend(metadata)

            top1 = correct / n

            # if hasattr(dataset, 'post_loop_metrics'):
            #     all_labels = torch.cat(all_labels)
            #     all_preds = torch.cat(all_preds)
            #     metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            #     if 'acc' in metrics:
            #         metrics['top1'] = metrics['acc']
            # else:
            metrics = {}

        if 'top1' not in metrics:
            metrics['top1'] = top1
        metrics['val_loss'] = val_loss
        # print(f"metrics={metrics}")

        if final_model:
            from stack import writeStackingResultsToCentralizedResultsFile
            runNum = args.results_db.split(".")[0].split("results")[-1] # e.g. results/results122.jsonl --> 122
            model_name = "Stack__"+"__".join(args.model_ckpts)+f"__{runNum}"
            writeAlphasToCentralizedAlphasFile(model_name,dataset_name,all_val_alphas)

        return metrics
    else:
        print(f"alphaModel.__class__.__name__ = {alphaModel.__class__.__name__}, expected 'ImageClassifier or ImageClassifier2'")


def evaluate3(model_ckpts, args):  # For 50-50 logit ensemble evaluation
    from stack import getLogitDataloader
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    preprocess_fn = models[0].val_preprocess
    if args.eval_datasets is None:
        return
    print("Starting evaluation on eval datasets.")
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        # print('Evaluating on', dataset_name)
        dataloader = getLogitDataloader(dataset_name,model_ckpts,preprocess_fn,args,is_train=False)

        results = eval_single_dataset3(dataloader, args)

        if 'top1' in results:
            sys.stdout.write(f"\t{dataset_name} Top-1 accuracy: {results['top1']:.4f}\n")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                sys.stdout.write(f"\t{dataset_name} {key}: {val:.4f}\n")
            info[dataset_name + ':' + key] = val
    if args.results_db is not None:
        if os.path.isfile(args.results_db):
            with open(args.results_db, 'r') as f:
                results = json.loads(f.read())
            results.append(info)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps(results))
        else:
            dirname = os.path.dirname(args.results_db)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps([info])) # List with info as its only element
            # print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def eval_single_dataset3(dataloader, args):  # For evaluation when stacking
    from stack import maybe_dictionarize2
    batched_data = enumerate(dataloader)
    
    device = args.device
    
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            sys.stdout.write(f"\r{i+1}/{len(dataloader)}")
            data = maybe_dictionarize2(data)  # Keys in dict: images, all_logits, labels
            # x = [data[key].cuda() for key in list(data.keys())[:-1]]
            all_logits = data['all_logits'].to(device)
            y = data['labels'].to(device)
            
            ensembled_logits = 0.5* all_logits[:,0] + 0.5* all_logits[:,1]
            
            pred = ensembled_logits.argmax(dim=1).to(device)
            
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            loss = loss_fn(ensembled_logits, y)
            val_loss += loss.item()

        top1 = correct / n
        
        metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    metrics['val_loss'] = val_loss
    return metrics

def eval_single_dataset_ose(models,dataset_name,args):   # Static equal-weight logit ('output=space') ensemble of 2 models on a dataset
    # Ideally, this should be able to handle equal-weight logit ensemble of >2 models, but this works for now
    preprocess_fn = models[0].val_preprocess
    data_location = args.data_location
    dataset, data_loader = get_dataloader_for_ose_or_oae(dataset_name,preprocess_fn,data_location)
    static_num_correct, static_total, static_total_loss = 0,0,0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    for batch in tqdm(data_loader):
        image,label = batch['images'],batch['labels']
        if hasattr(dataset, 'project_labels'):
            label = dataset.project_labels(label, args.device)
        image,label,model1,model2 = image.to(args.device),label.to(args.device),models[0].to(args.device),models[1].to(args.device)
        logits1,logits2 = model1(image),model2(image)
        
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits1,logits2 = projection_fn(logits1, args.device),projection_fn(logits2, args.device)

        static_loss,static_correct = useLogitEnsemble(label,loss_fn,logits1,logits2,dataset,args,alpha=0.5)
        static_num_correct += (1 if static_correct == True else 0) # batch_size is 1
        static_total += label.size(0)
        static_total_loss += static_loss
    accuracy = static_num_correct / static_total
    val_loss = static_total_loss / static_total
    return accuracy, val_loss

def eval_single_dataset_oae(models,dataset_name,args):   # Static optimal alpha (logit) ensemble of 2 models on a dataset
    # Ideally, this should be able to handle optimal alpha ensemble of >2 models, but this works for now
    preprocess_fn = models[0].val_preprocess
    data_location = args.data_location
    dataset, data_loader = get_dataloader_for_ose_or_oae(dataset_name,preprocess_fn,data_location)
    num_correct, total, total_loss = 0,0,0
    optimalAlphas = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    for batch in tqdm(data_loader):
        image,label = batch['images'],batch['labels']
        if hasattr(dataset, 'project_labels'):
            label = dataset.project_labels(label, args.device)
        image,label,model1,model2 = image.to(args.device),label.to(args.device),models[0].to(args.device),models[1].to(args.device)
        logits1,logits2 = model1(image),model2(image)
        
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits1,logits2 = projection_fn(logits1, args.device),projection_fn(logits2, args.device)

        optimalAlpha = getOptimalAlpha(logits1,logits2,label,loss_fn,args)
        optimalAlphas.append(optimalAlpha.item())

        loss,correct = useLogitEnsemble(label,loss_fn,logits1,logits2,dataset,args,alpha=optimalAlpha)
        num_correct += (1 if correct == True else 0) # batch_size is 1
        total += label.size(0)
        total_loss += loss
    accuracy = num_correct / total
    val_loss = total_loss / total
    return accuracy, val_loss, optimalAlphas

if __name__ == "__main__":
    # This is intended to compute the accuracy with a 50-50 logit ensemble, but it doesn't work yet
    from src.args2 import parse_arguments2
    args = parse_arguments2()
    evaluate3(args.model_ckpts, args)