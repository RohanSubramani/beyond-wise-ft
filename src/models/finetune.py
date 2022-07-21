import os
import copy
import time
import tqdm
import sys
import wandb

import torch

import clip.clip as clip

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing

import src.datasets as datasets


def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    
    
    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.train_preprocess # val_preprocess? Idk why it started with that
        image_enc = image_classifier.image_encoder
        # print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        # print_every = 1000
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    wandb.init(project="CLIP-transfer-learning",entity="rohansubramani",name=args.save)
    # # Name is now ckptNum b/c there are too many args, and you can check config in wandb, or ckpt_and_run_details.txt.
    wandb.config.update(args)
    wandb.config.net_name = "CLIP-ViT-B-32"
    wandb.watch(model)

    # Evaluate with the zeroshot model
    args.current_epoch = -1
    t0 = time.time()
    eval_results = evaluate(image_classifier, args)
    t1 = time.time()
    eval_time = t1-t0
    wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)
        
        args.current_epoch = epoch

        if epoch == 0:
            print_every = (len(data_loader)*(args.epochs) // 25)+1 if eval_time<100 else (len(data_loader)*(args.epochs) // 10)+1
            if print_every < 75:
                print_every = 75  # Don't want to be evaluating too often in a small run
            print(f"Evaluates every {print_every} batches.")
            j=0 # Keeps track of total batches completed

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            
            logits = model(inputs) # This line caused stopping the first time I tried on gpu 5

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            percent_complete = 100 * i / len(data_loader)
            details = f"\rTrain Epoch: {epoch+1}/{args.epochs} [{percent_complete:.0f}% {i}/{len(data_loader)}]\t"+\
                      f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
            
            sys.stdout.write(details)
            
            if (j+1) % print_every == 0:
                print("")
                eval_results = evaluate(image_classifier, args)
                wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})
            
            j += 1  # Unlike i, j doesn't reset at the end of each epoch
        
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None and (epoch%((args.epochs//5)+1)==0 or epoch+1 == args.epochs):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            image_classifier.save(model_path)
            print('\nSaved model to', model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)
    
    print("")
    eval_results = evaluate(image_classifier, args)
    wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})
    
    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)