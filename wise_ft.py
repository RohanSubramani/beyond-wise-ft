import os

import numpy as np

import torch

import json

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from src.plot import makePlots

def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def wise_ft(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    if args.load is None:
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned')
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    if args.results_db is not None:
        if os.path.isfile(args.results_db):
            with open(args.results_db, 'r') as f:
                results = json.loads(f.read())
                zeroshot_results = results[0]
                finetuned_results = results[-1]

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))
        
        print(f"alpha={alpha}")


        # This assumes something was passes in for results_db! If not, you can resume from after finetuning, and remove the if & elif

        if alpha == 0.0:
            zeroshot_results["alpha"], zeroshot_results["current_epoch"] = 0.0, args.epochs-1
            for dataset in args.eval_datasets:
                key = dataset+":top1"
                print(f"{dataset} Acc: {zeroshot_results[key]}")
            with open(args.results_db, 'r') as f:
                results = json.loads(f.read())
            results.append(zeroshot_results)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps(results))

        elif alpha == 1.0:
            finetuned_results["alpha"], finetuned_results["current_epoch"] = 1.0, args.epochs-1
            for dataset in args.eval_datasets:
                key = dataset+":top1"
                print(f"{dataset} Acc: {finetuned_results[key]}")
            with open(args.results_db, 'r') as f:
                results = json.loads(f.read())
            results.append(finetuned_results)
            with open(args.results_db, 'w') as f:
                f.write(json.dumps(results))

        else:
            # evaluate
            evaluate(finetuned, args)

    makePlots(args.results_db,args.eval_datasets,alphas,args.save)

if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)