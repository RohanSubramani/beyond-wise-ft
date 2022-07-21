# This is experimental code, not used for anything.

import json
import matplotlib.pyplot as plt
import os
import numpy as np

results_db = "./results_test.jsonl"

for i in range(2):
    info = {i:np.sin(i)}
    if results_db is not None:
        dirname = os.path.dirname(results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(results_db, 'a+') as f:
            try:
                results = json.load(results_db)
            except json.decoder.JSONDecodeError:
                print("Starting")
                results = []
        results.append(info)
        with open(results_db, 'w') as f:
            f.write(json.dumps(results))

ensemble_results = [result for result in results[-11:]]

for dataset in ["ImageNet","ImageNetV2","ImageNetR","ImageNetSketch"]:
    xs = [0.1*x for x in range(11)]
    ys = [result[dataset+":top1"] for result in ensemble_results]
    plt.scatter(xs,ys,label=dataset)
    # plt.show()
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plotTest")
print("Done.")





#################################################################################################################



# Full version of eval.py that I had in mind for incorporating the plotting technique above.


import os
import json
import sys

import torch
import numpy as np

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize

import src.datasets as datasets


def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

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
        if os.isfile(args.results_db):
            with open(args.results_db, 'r') as f:
                results = json.loads(f)
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