import torch
import wandb
import time
import sys
import os
import json

from src.args2 import parse_arguments2
from src.models.modeling import * # Includes ImageClassifier, ResNet, ClassificationHead2
import src.datasets as datasets
from src.datasets.common import get_dataloader
from open_clip.src.open_clip.transform import image_transform
from src.datasets.common import FeatureDataset,maybe_dictionarize
from torch.utils.data import TensorDataset, DataLoader
from src.models.zeroshot import get_zeroshot_classifier
from src.models.utils import cosine_lr, LabelSmoothing
from src.models.eval import evaluate2


def trainAlphaModel(args):
    num_models = len(args.model_ckpts)
    model, alphaModel, preprocess_fn, image_enc = getAlphaModel(args,num_models)  # This is not general: for non-CLIP alphaModels, you shouldn't be getting the preprocess function from here?
    # args.model_class,args.model_details
    data_loader = getLogitDataloader(args.train_dataset,args.model_ckpts,preprocess_fn,args)
    train(model,alphaModel,data_loader,image_enc,args)


def getAlphaModel(args,num_models):  # model_class,model_details
    if args.load is not None:
        image_classifier = ImageClassifier2.load(args.load)  # args.load is here for the alpha model ckpt

        if args.freeze_encoder:
            print('Fine-tuning a linear classifier')
            # image_encoder = ImageEncoder(args, keep_lang=True)
            # classification_head = get_zeroshot_classifier(args, image_encoder.model)
            image_classifier.classification_head = ClassificationHead2(normalize=True, input_size=512, output_size=num_models, biases=None) 
            # .image_encoder.model
            model = image_classifier.classification_head
            # input_key = 'features'
            preprocess_fn = image_classifier.val_preprocess # not train_preprocess, because data aug isn't needed if learned features are fixed
            image_enc = image_classifier.image_encoder
        else:
            print('Fine-tuning end-to-end')
            model = image_classifier
            # input_key = 'images'
            preprocess_fn = image_classifier.train_preprocess
            cont = int(input("Are you sure that using train preprocess is ok? This likely includes data aug, which makes it impossible to store logits for all images and ensemble them. 1 for yes, continue. 2 for no, stop."))
            assert cont == 1, "You chose to stop because train preprocess (and likely data aug) is being used "
            image_enc = torch.nn.Identity()
            image_classifier.process_images = True

        return model, image_classifier, preprocess_fn, image_enc # , input_key


def getLogitDataloader(dataset_name,model_ckpts,preprocess_fn,args,is_train=True):
    get_logit_dataloader_fxn_dict = {'DeterministicImageNet':getDeterministicImageNetLogitDataloader,'ImageNetV2':getImageNetV2LogitDataloader,'ImageNetR':getImageNetRLogitDataloader,'ImageNetSketch':getImageNetSketchLogitDataloader,'ImageNetA':getImageNetALogitDataloader,'ObjectNet':getObjectNetLogitDataloader,'CIFAR10':getCIFAR10LogitDataloader,'CIFAR101':getCIFAR101LogitDataloader,'CIFAR102':getCIFAR102LogitDataloader}

    get_logit_dataloader_fxn = get_logit_dataloader_fxn_dict[dataset_name]
    logit_dataloader = get_logit_dataloader_fxn(model_ckpts,preprocess_fn,args,is_train)
    
    return logit_dataloader

def train(model,alphaModel,data_loader,image_enc,args): # input_key
    
    device = args.device
    for item in [model, image_enc, alphaModel]:
        item.to(device)
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    num_batches = len(data_loader)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    wandb.init(project="stacking-experiments",entity="rohansubramani",name=args.save)
    # # Name is now ckptNum b/c there are too many args, and you can check config in wandb, or ckpt_and_run_details.txt.
    wandb.config.update(args)
    wandb.config.net_name = "CLIP-ViT-B-32"
    wandb.watch(model)

    args.current_epoch = -1
    t0 = time.time()
    eval_results = evaluate2(alphaModel, args.model_ckpts, args)
    t1 = time.time()
    eval_time = t1-t0
    wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})

    for epoch in range(args.epochs):
        for item in [model, image_enc, alphaModel]:
            item.to(device)
        model.train()
        
        args.current_epoch = epoch

        if epoch == 0:
            print_every = (len(data_loader)*(args.epochs) // 25)+1 if eval_time<100 else (len(data_loader)*(args.epochs) // 10)+1
            if print_every < 75:
                print_every = 75  # Don't want to be evaluating too often in a small run
            print(f"Evaluates every {print_every} batches.")
            j=0 # Keeps track of total batches completed

        i=0
        for batch in data_loader:  # This is where subsetting leads to issues, due to indexing mismatch
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize2(batch)
            inputs = [image_enc(batch['images'].cuda()), batch['all_logits'].cuda()]
            labels = batch['labels'].cuda()
            
            data_time = time.time() - start_time

            logits = model(*inputs) # This line caused stopping the first time I tried on gpu 5

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
                eval_results = evaluate2(alphaModel, args.model_ckpts, args)
                wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})
            
            i += 1
            j += 1  # Unlike i, j doesn't reset at the end of each epoch
        
        if args.freeze_encoder:
            alphaModel = ImageClassifier2(alphaModel.image_encoder, model.module)
        else:
            alphaModel = model.module

        # Saving model
        if args.save is not None and (epoch%((args.epochs//5)+1)==0 or epoch+1 == args.epochs):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            alphaModel.save(model_path)
            print('\nSaved model to', model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)
    
    print("")
    eval_results = evaluate2(alphaModel, args.model_ckpts, args)
    wandb.log({dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets})
    
    if args.save is not None:
        return model_path

def maybe_dictionarize2(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 3:
        batch = {'images': batch[0], 'all_logits': batch[1], 'labels': batch[2]}
    if len(batch) == 4:
        batch = {'images': batch[0], 'all_logits': batch[1], 'labels': batch[2], 'image_paths': batch[3]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}. Expected 3 or 4.')

    return batch


#####       Experimental code       #####

def getDeterministicImageNetLogitDataloader(model_ckpts,preprocess_fn,args,is_train=True):  # original = "without logits"    
    dataset_class = getattr(datasets, 'DeterministicImageNet')
    logit_dataset_class = getattr(datasets, 'DeterministicImageNetWithLogits')
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    model_names = [model_ckpt.split("wiseft/")[-1].split(".")[0] for model_ckpt in model_ckpts]
    # Ex. "./models/wiseft/ViTB32_8/zeroshot.pt" --> ViTB32_8/zeroshot
    logit_datasets = [FeatureDataset(is_train=is_train, image_encoder=models[i], dataset=dataset, device=args.device,\
        model_name=model_names[i]).data['logits'] for i in range(len(models))]  # Not actually using image encoders to get features, using
    # models to get logits
    if is_train:
        print(f"Producing all_logits_dataset for DeterministicImageNet, will take a few minutes. (~4? Mostly for converting to tensor.)")
    all_logits_dataset = torch.tensor([[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))],dtype=torch.float32)
    finalDataset = logit_dataset_class(preprocess_fn,all_logits_dataset,location=args.data_location,batch_size=args.batch_size,
        subset_proportion=args.subset_proportion, is_train=is_train)
    data_loader = finalDataset.train_loader if is_train else finalDataset.test_loader
    return data_loader

def getImageNetV2LogitDataloader(model_ckpts,preprocess_fn,args,*more_args,**kwargs):  # original = "without logits"
    # *more_args,**kwargs is included to absorb additional inputs that aren't relevant here, like "is_train"
    dataset_class = getattr(datasets, 'ImageNetV2')
    logit_dataset_class = getattr(datasets, 'ImageNetV2WithLogits')
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
    model_names = [model_ckpt.split("wiseft/")[-1].split(".")[0] for model_ckpt in model_ckpts]
    # Ex. "./models/wiseft/ViTB32_8/zeroshot.pt" --> ViTB32_8/zeroshot
    logit_datasets = [FeatureDataset(is_train=False, image_encoder=models[i], dataset=dataset, device=args.device,\
        model_name=model_names[i]).data['logits'] for i in range(len(models))]  # Not actually using image encoders to get features, using
    # models to get logits
    all_logits_dataset = torch.tensor([[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))],dtype=torch.float32)
    finalDataset = logit_dataset_class(preprocess_fn,all_logits_dataset,location=args.data_location,batch_size=args.batch_size)
    data_loader = finalDataset.test_loader
    return data_loader

def getImageNetRLogitDataloader(model_ckpts,preprocess_fn,args,*more_args,**kwargs):  # original = "without logits"
    pass

def getImageNetSketchLogitDataloader():
    pass

def getImageNetALogitDataloader():
    pass

def getObjectNetLogitDataloader():
    pass

def getCIFAR10LogitDataloader():
    pass

def getCIFAR101LogitDataloader():
    pass

def getCIFAR102LogitDataloader():
    pass


if __name__ == "__main__":
    print("\n\nMAKE SURE THAT YOUR TRAIN LOADER IS DETERMINISTIC! (Otherwise logit ensembling won't work at all.)\n\n")
    args = parse_arguments2()
    trainAlphaModel(args)

#####            Earlier Code             #####
    
# def getDataset(original_dataset,model_ckpts,preprocess_fn,args,is_train=True):  # original = "without logits"
#     corr_logits = {'DeterministicCIFAR10':'DeterministicCIFAR10WithLogits','DeterministicImageNet':'DeterministicImageNetWithLogits'}    
#     dataset_class = getattr(datasets, original_dataset)
#     logit_dataset_class = getattr(datasets, corr_logits[original_dataset])
#     # t0 = time.time()
#     dataset = dataset_class(
#         preprocess_fn,
#         location=args.data_location,
#         batch_size=args.batch_size
#     )
#     # t1=time.time()
#     # print(f"Time to get dataset = {t1-t0}")
#     # print(f"model_ckpts={model_ckpts}")
#     # time.sleep(5)
#     models = [ImageClassifier.load(ckpt) for ckpt in model_ckpts]
#     model_names = [model_ckpt.split("wiseft/")[-1].split(".")[0] for model_ckpt in model_ckpts]
#     # Ex. "./models/wiseft/ViTB32_8/zeroshot.pt" --> ViTB32_8/zeroshot
#     # print(f"dir(dataset):\n{dir(dataset)}")
#     # print(dataset.batch_size)
#     logit_datasets = [FeatureDataset(is_train=is_train, image_encoder=models[i], dataset=dataset, device=args.device, model_name=model_names[i]).data['logits'] for i in range(len(models))]  # Not actually using image encoders to get features, using
#     # models to get logits
#     # print(f"logit_datasets[0].shape={logit_datasets[0].shape}")
#     if is_train:
#         print(f"Producing all_logits_dataset, may take a few minutes for large datasets (~4 minutes for ImageNet) (mostly for converting to tensor).")
#     all_logits_dataset = torch.tensor([[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))],dtype=torch.float32)

#     # print(f"all_logits_dataset[0].shape={all_logits_dataset[0].shape}")
#     finalDataset = logit_dataset_class(preprocess_fn,all_logits_dataset,location=args.data_location,batch_size=args.batch_size,
#         subset_proportion=args.subset_proportion, is_train=is_train)
#     data_loader = finalDataset.train_loader if is_train else finalDataset.test_loader
#     # data_loader = DataLoader(finalDataset, batch_size=args.batch_size, shuffle=False)
#     return data_loader