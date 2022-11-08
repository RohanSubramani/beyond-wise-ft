import torch
import wandb
import time
import sys
import os
import math
import pandas as pd
import json
import copy
from torch.nn import DataParallel

from src.args2 import parse_arguments2
from src.models.modeling import ImageClassifier, ClassificationHead2
from src.models.utils import getOptimizer
import src.datasets as datasets
# from src.datasets.common import get_dataloader
# from open_clip.src.open_clip.transform import image_transform
from src.datasets.common import FeatureDataset #,maybe_dictionarize
# from torch.utils.data import TensorDataset, DataLoader
# from src.models.zeroshot import get_zeroshot_classifier
from src.models.utils import cosine_lr, LabelSmoothing
from src.models.eval import evaluate2, eval_single_dataset, eval_single_dataset_ose, eval_single_dataset_oae


def trainAlphaModel(args):
    num_models = len(args.model_ckpts)
    model, alphaModel, preprocess_fn, image_enc = getAlphaModel(args,num_models)  # This is not general: for non-CLIP alphaModels, you shouldn't be getting the preprocess function from here?
    # args.model_class,args.model_details
    data_loader = getLogitDataloader(args.train_dataset,args.model_ckpts,preprocess_fn,args)
    eval_results = train(model,alphaModel,data_loader,image_enc,args)
    writeStackingResultsToCentralizedResultsFile(eval_results,args)
    saveComparisons(args)

def getAlphaModel(args,num_models):  # model_class,model_details
    if args.load is not None:
        image_classifier = ImageClassifier.load(args.load)  # args.load is here for the alpha model ckpt

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
            image_classifier.process_images = False  # FIXME This probably will raise an error. Need to change process_images after creating logit dataset 

        assert image_classifier.classification_head.__class__.__name__ == "ClassificationHead2", f"image_classifier.classification_head.__class__.__name__ should be ClassificationHead2 to ensure that the softmax occurs within the alpha model. Is currently {image_classifier.classification_head.__class__.__name__}."

        return model, image_classifier, preprocess_fn, image_enc # , input_key


def getLogitDataloader(dataset_name,model_ckpts,preprocess_fn,args,is_train=True):
    get_logit_dataloader_fxn_dict = {'DeterministicImageNet':getDeterministicImageNetLogitDataloader,'ImageNetV2':getImageNetV2LogitDataloader,'ImageNetR':getImageNetRLogitDataloader,'ImageNetSketch':getImageNetSketchLogitDataloader,'ImageNetA':getImageNetALogitDataloader,'ObjectNet':getObjectNetLogitDataloader,'CIFAR10':getCIFAR10LogitDataloader,'CIFAR101':getCIFAR101LogitDataloader,'CIFAR102':getCIFAR102LogitDataloader}
    
    print(f"Getting logit dataloader for {dataset_name}.")
    t0 = time.time()
    get_logit_dataloader_fxn = get_logit_dataloader_fxn_dict[dataset_name]
    logit_dataloader = get_logit_dataloader_fxn(model_ckpts,preprocess_fn,args,is_train)
    t1 = time.time()
    obtain_time = t1 - t0
    print(f"{dataset_name} dataloader obtained in {obtain_time} seconds.")

    return logit_dataloader

def train(model,alphaModel,data_loader,image_enc,args): # input_key
    
    # alphaModel.process_images = False

    device = args.device

    for item in [image_enc, alphaModel]:
        item.to(device) # These actually do work and are necessary, I wasn't sure originally

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = DataParallel(model, device_ids=devices)
    # image_enc = DataParallel(image_enc, device_ids=devices)
    # alphaModel = DataParallel(alphaModel, device_ids=devices) 
    model.train()

    num_batches = len(data_loader)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getOptimizer(params,args) # AdamW or SGD, with specified learning rate and weight decay

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
    args.eval_num += 1
    eval_time = t1-t0
    log_dict = {dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets}
    for dataset in args.eval_datasets:
        log_dict[dataset+" Val Loss"] = eval_results[dataset+":val_loss"]
    wandb.log(log_dict)

    total_num_batches = num_batches*args.epochs
    print_every = math.ceil(total_num_batches / 25) if eval_time<100 else math.ceil(total_num_batches / 10)
    if print_every < 75:
        print_every = 75  # Don't want to be evaluating too often in a small run
    print(f"Evaluates every {print_every} batches.")
    # There is more adjustment with print_every below, because evaluating too frequently takes an excessive amount of time, but having 
    # enough evaluations to have a sense of what's going on during training is important.
    
    j=0 # Keeps track of total batches completed
    total_loss = 0  # Resets to zero after a number of batches equal to print_every.
    total_grad_norm = 0 # Keeps track of sum of gradient norms between evaluations of average gradient norm (for checking convergence)
    total_average_alpha = 0 # Sum of (average alpha this batch) across all batches since last print_every 
    
    for epoch in range(args.epochs):
        for item in [model, image_enc, alphaModel]:
            item.to(device) # These actually do work and are necessary, I wasn't sure originally

        # model = DataParallel(model, device_ids=devices)
        # image_enc = DataParallel(image_enc) #, device_ids=devices)
        # alphaModel = DataParallel(alphaModel) # , device_ids=devices) 
        model.train()
        
        args.current_epoch = epoch

        i=0
        for batch in data_loader:  # This is where subsetting leads to issues, due to indexing mismatch
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize2(batch)
            images = image_enc(batch['images'].to(device)) # images = image_enc(batch['images'].cuda())
            all_logits = batch['all_logits'].to(device)

            labels = batch['labels'].to(device)
            
            data_time = time.time() - start_time

            alphas = alphaModel(images)

            model1_alphas = alphas[:,0]
            average_alpha = torch.mean(model1_alphas)
            total_average_alpha += average_alpha
                
            # logits = model(*inputs) # This line caused stopping the first time I tried on gpu 5            
            # alphas.shape = torch.Size([batch_size, batch_size, num_models])  # Surely this is all_logits??? I hope so.
            
            weird_ensembled_logits = alphas @ all_logits
            fixed_ensembled_logits = torch.transpose(torch.diagonal(weird_ensembled_logits),0,1)
            # This works, but it's messy. Rohan Subramani has the evidence in favor of this working in a screenshot of a Colab notebook in a Google doc

            loss = loss_fn(fixed_ensembled_logits, labels)
            total_loss += loss.item()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            total_grad_norm += grad_norm

            optimizer.step()
            batch_time = time.time() - start_time

            percent_complete = 100 * i / len(data_loader)
            details = f"\rTrain Epoch: {epoch+1}/{args.epochs} [{percent_complete:.0f}% {i}/{len(data_loader)}]\t"+\
                      f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
            
            sys.stdout.write(details)

            # If validation checkpoints are taking more than a tenth of the training time, and the total validation time is on track to
            # take over 40 mins, check validation accuracy less frequently.
            train_to_eval_ratio = (batch_time * print_every) / eval_time
            predicted_total_eval_time = (total_num_batches / print_every) * eval_time
            if train_to_eval_ratio < 9 and predicted_total_eval_time > 2400:
                print_every = math.ceil(9*eval_time/batch_time)
                print(f"Evaluates every {print_every} batches.")
                # This makes it so that evaluation checkpoints take <= one tenth of total training time, and train_to_eval_ratio >= 9.
            
            if (j+1) % print_every == 0:
                eval_start_time = time.time()
                print("")
                eval_results = evaluate2(alphaModel, args.model_ckpts, args)
                args.eval_num += 1
                log_dict = {dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets}
                
                for dataset in args.eval_datasets:
                    log_dict[dataset+" Val Loss"] = eval_results[dataset+":val_loss"]

                log_dict["Train Loss"] = total_loss / print_every
                log_dict["Average Alpha"] = total_average_alpha / print_every
                log_dict["Gradient Norm"] = total_grad_norm / print_every
                
                # if args.compare_with_optimal_alphas:
                #     optimal_alphas = get_optimal_alphas(logits1,logits2,labels,loss_fn,args)
                #     plot(model1_alphas,optimal_alphas,plot_file,j+1)
                
                wandb.log(log_dict)
                total_loss,total_average_alpha,total_grad_norm = 0,0,0
                eval_end_time = time.time()
                eval_time = eval_end_time - eval_start_time
            
            i += 1
            j += 1  # Unlike i, j doesn't reset at the end of each epoch
        
        if args.freeze_encoder:
            alphaModel = ImageClassifier(alphaModel.image_encoder, model.module) # Previously ImageClassifier2, that became irrelevant
        else:
            alphaModel = model.module
        
        assert alphaModel.classification_head.__class__.__name__ == "ClassificationHead2", f"alphaModel.classification_head.__class__.__name__ should be ClassificationHead2 to ensure that the softmax occurs within the alpha model. Is currently {alphaModel.classification_head.__class__.__name__}"

        # Saving model
        if args.save is not None and (epoch%((args.epochs//5)+1)==0 or epoch+1 == args.epochs):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            alphaModel.save(model_path)
            print('\nSaved model to', model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)
    
    print("")
    eval_results = evaluate2(alphaModel, args.model_ckpts, args, final_model=True)

    log_dict = {dataset+" Acc": eval_results[dataset+":top1"] for dataset in args.eval_datasets}
    for dataset in args.eval_datasets:
        log_dict[dataset+" Val Loss"] = eval_results[dataset+":val_loss"]
    wandb.log(log_dict)

    return eval_results

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

def getDeterministicImageNetLogitDataloader(model_ckpts,preprocess_fn,args,is_train=True): 
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
    
    if is_train:
        print("Producing all_logits_dataset for DeterministicImageNet, will take a few minutes. (~4? Mostly for converting to tensor.)")
    
    logit_datasets = torch.tensor([FeatureDataset(is_train=is_train, image_encoder=models[i], dataset=dataset, device=args.device,\
        model_name=model_names[i]).data['logits'] for i in range(len(models))])  # Not actually using image encoders to get features, using
    # models to get logits

    # logit_datasets.shape = 2 x n (eg 1.3 mil) x 1000
    
    print(f"logit_datasets.shape = {logit_datasets.shape}")
    all_logits_dataset = logit_datasets.permute(1, 0, 2)
    print(f"all_logits_dataset.shape = {all_logits_dataset.shape}")
        
    # all_logits_dataset = [[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))]
    # for logit_pair in all_logits_dataset:
    #     logit_pair[1] *= -1
    
    # all_logits_dataset = torch.tensor([[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))],dtype=torch.float32)

    if args.diagnostic_test:
        all_logits_dataset[:,0] *= -1   
        # Just for alpha model test, sabotage one model by negating its logits, alpha model should learn not to use those logits 

    finalDataset = logit_dataset_class(preprocess_fn,all_logits_dataset,location=args.data_location,batch_size=args.batch_size,
        subset_proportion=args.subset_proportion, is_train=is_train)
    data_loader = finalDataset.train_loader if is_train else finalDataset.test_loader
    return data_loader

def getImageNetV2LogitDataloader(model_ckpts,preprocess_fn,args,*more_args,**kwargs):  
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
    all_logits_dataset = torch.tensor([[logit_dataset[i] for logit_dataset in logit_datasets] for i in range(len(logit_datasets[0]))],
    dtype=torch.float32)

    if args.diagnostic_test:
        all_logits_dataset[:,0] *= -1   # Just for alpha model test

    finalDataset = logit_dataset_class(preprocess_fn,all_logits_dataset,location=args.data_location,batch_size=args.batch_size)
    data_loader = finalDataset.test_loader
    return data_loader

def getImageNetRLogitDataloader(model_ckpts,preprocess_fn,args,*more_args,**kwargs): 
    # *more_args,**kwargs is included to absorb additional inputs that aren't relevant here, like "is_train"
    dataset_class = getattr(datasets, 'ImageNetR')
    logit_dataset_class = getattr(datasets, 'ImageNetRWithLogits')
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

def writeStackingResultsToCentralizedResultsFile(eval_results,args):
    centralized_results_file = "./central_results.txt"
    runNum = args.results_db.split(".")[0].split("results")[-1] # e.g. results/results122.jsonl --> 122
    model_name = "Stack__"+"__".join(args.model_ckpts)+f"__{runNum}"
    for eval_dataset in args.eval_datasets:
        key = f"{model_name}, {eval_dataset}"
        with open(centralized_results_file,"r") as file:
            results = json.loads(file.read())
            results[key]["accuracy"] = eval_results[eval_dataset+":top1"]
            results[key]["val_loss"] = eval_results[eval_dataset+":val_loss"]
        with open(centralized_results_file,"w") as file: # Overwrites previous file with the new additions
            json.dump(results,file)

def saveComparisons(args):
    model_names = args.model_ckpts
    model_names.append("WSE__"+"__".join(args.model_ckpts)) # Weight space ensemble
    model_names.append("OSE__"+"__".join(args.model_ckpts)) # Output space ensemble
    model_names.append("OAE__"+"__".join(args.model_ckpts)) # Optimal alpha ensemble
    runNum = args.results_db.split(".")[0].split("results")[-1] # e.g. results/results122.jsonl --> 122
    model_names.append("Stack__"+"__".join(args.model_ckpts)+f"__{runNum}") # Stacking ensemble
    abbreviated_model_names = [f"model{i+1}" for i in range(len(args.model_ckpts))]+["WSE","OSE","OAE","Stack"]

    accuracy_table = [[0 for j in model_names] for i in args.eval_datasets]
    val_loss_table = [[0 for j in model_names] for i in args.eval_datasets]
    for i in range(len(args.eval_datasets)):
        dataset = args.eval_datasets[i]
        for j in range(len(model_names)):
            accuracy_table[i][j],val_loss_table[i][j] = __get_accuracy_and_val_loss(model_names[j],dataset,args)
            
            # Save after each addition, so even if something goes wrong, what is completed so far will be saved.
            accuracy_df = pd.DataFrame(accuracy_table, args.eval_datasets, abbreviated_model_names)
            val_loss_df = pd.DataFrame(val_loss_table, args.eval_datasets, abbreviated_model_names)
            if args.save is not None:
                os.makedirs(args.save, exist_ok=True)
                accuracy_df.to_csv(os.path.join(args.save, f'accuracy_table.csv')) # Overwrites existing file
                val_loss_df.to_csv(os.path.join(args.save, f'val_loss_table.csv')) # Overwrites existing file
        
def __get_accuracy_and_val_loss(model_name,eval_dataset,args):
    centralized_results_file = "./central_results.txt"
    key = f"{model_name}, {eval_dataset}"
    try:
        with open(centralized_results_file,"r") as file:
            results = json.loads(file.read())
            if key in results.keys():
                accuracy = results[key]["accuracy"]
                val_loss = results[key]["val_loss"]
            else:
                accuracy,val_loss = __compute_accuracy_and_val_loss(model_name,eval_dataset,args)
                results[key]["accuracy"] = accuracy
                results[key]["val_loss"] = val_loss
    except FileNotFoundError:   # What if the file is empty, and results.keys() isn't available? I think that will never happen though.
        print("Caught FileNotFoundError.")
        accuracy,val_loss = __compute_accuracy_and_val_loss(model_name,eval_dataset,args)
        results = {}
        results[key] = {}
        results[key]["accuracy"] = accuracy
        results[key]["val_loss"] = val_loss
    with open(centralized_results_file,"w") as file:
        json.dump(results,file)
    return accuracy,val_loss

def __compute_accuracy_and_val_loss(model_name,eval_dataset,args):
    parsing = model_name.split("__")

    if len(parsing) == 1: # If this is one of the base model ckpts, rather than an ensemble
        model = ImageClassifier.load(model_name)
        metrics = eval_single_dataset(model, eval_dataset, args)
        accuracy,val_loss = metrics['top1'],metrics['val_loss']
    
    elif parsing[0]=="WSE":
        models = [ImageClassifier.load(model_name) for model_name in parsing[1:]]
        wse_model = getStaticWSEModel(models)
        metrics = eval_single_dataset(wse_model, eval_dataset, args)
        accuracy,val_loss = metrics['top1'],metrics['val_loss']

    elif parsing[0]=="OSE":
        models = [ImageClassifier.load(model_name) for model_name in parsing[1:]]
        accuracy,val_loss = eval_single_dataset_ose(models,eval_dataset,args)
        
    elif parsing[0]=="OAE":
        models = [ImageClassifier.load(model_name) for model_name in parsing[1:]]
        accuracy,val_loss,optimalAlphas = eval_single_dataset_oae(models,eval_dataset,args)
        writeAlphasToCentralizedAlphasFile(model_name,eval_dataset,optimalAlphas)

    # elif parsing[0]=="Stack": Don't need to worry about this case, the way saveComparisons is currently called this never happens.
    # However, this is a strange asymmetrical treatment of stacking, and (TODO) I hope to fix the symmetry in the future by making
    # saveComparisons the "primary" function, rather than trainAlphaModel. So saveComparisons can call trainAlphaModel if it looks for 
    # stacking results and doesn't find them.

    else:
        raise ValueError(f"parsing[0] (of model name) = {parsing[0]}, should be in ['WSE','OSE','OAE'] or be a single model name.")
        
    return accuracy,val_loss

def getStaticWSEModel(models,alphas=None):
    assert alphas is None or len(alphas)==len(models) or len(alphas)==len(models)-1, f"Should satisy 'alphas is None or len(alphas)==len(models) or len(alphas)==len(models)-1', but len(alphas)={len(alphas)} and len(models)={len(models)}."
    if alphas is None:
        alphas = [1/len(models) for i in range(len(models))]
    if len(alphas)==len(models)-1:
        alphas.append(1-sum(alphas))
    for alpha in alphas:
        assert alpha <= 1 and alpha >= 0, f"alpha = {alpha}, should be between 0 and 1, inclusive."
    assert abs(sum(alphas)-1) < 0.001, f"sum(alphas)-1={sum(alphas)-1}, should be 0."
    thetas = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    for i in range(len(thetas)):
        assert set(thetas[0].keys()) == set(thetas[i].keys())

        ## Make it equal weighting if alphas is none, or use alphas for weighting
    theta_new = {key: sum([alphas[i]*thetas[i][key] for i in range(len(alphas))]) for key in thetas[0].keys()}

    static_wse_model = copy.deepcopy(models[0])
    static_wse_model.load_state_dict(theta_new)
    return static_wse_model

def writeAlphasToCentralizedAlphasFile(model_name,eval_dataset,alphas):
    centralized_alphas_file = "./central_alphas.txt"
    key = f"{model_name}, {eval_dataset}"
    try:
        with open(centralized_alphas_file,"r") as file:
            results = json.loads(file.read())
            if key in results.keys():
                return
            else:
                results[key] = alphas
    except FileNotFoundError:   # What if the file is empty, and results.keys() isn't available? I think that will never happen though.
        print("Caught FileNotFoundError.")
        results = {}
        results[key] = alphas
    with open(centralized_alphas_file,"w") as file:
        json.dump(results,file)

if __name__ == "__main__":
    # print("\n\nMAKE SURE THAT YOUR TRAIN LOADER IS DETERMINISTIC! (Otherwise logit ensembling won't work at all.)\n\n")
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