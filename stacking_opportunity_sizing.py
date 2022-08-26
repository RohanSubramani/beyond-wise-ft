import torch
import src.datasets as datasets
import json
import argparse
import os
from tqdm import tqdm
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from src.models.modeling import ImageClassifier

def sizeOpportunity(model1,model2,dataset_name,args):
    preprocess_fn = model1.val_preprocess
    data_location = args.data_location
    t0 = time.time()
    dataset, data_loader = getDataloader(dataset_name,preprocess_fn,data_location)
    t1=time.time()
    print(f"Time to get dataloader: {t1-t0}.")
    results = {}
    num_correct, total = 0,0
    static_num_correct, static_total = 0,0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    i=0
    print_after = 200
    for batch in tqdm(data_loader):
        image,label = batch['images'],batch['labels']
        if hasattr(dataset, 'project_labels'):
            print('0')
            label = dataset.project_labels(label, args.device)
        image,label,model1,model2 = image.to(args.device),label.to(args.device),model1.to(args.device),model2.to(args.device)
        logits1,logits2 = model1(image),model2(image)
        
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits1,logits2 = projection_fn(logits1, args.device),projection_fn(logits2, args.device)

        optimalAlpha = getOptimalAlpha(logits1,logits2,label,loss_fn,args)
        loss,correct = useLogitEnsemble(label,loss_fn,logits1,logits2,dataset,args,alpha=optimalAlpha)
        num_correct += (1 if correct == True else 0)
        total += label.size(0)

        static_loss,static_correct = useLogitEnsemble(label,loss_fn,logits1,logits2,dataset,args,alpha=0.5)
        static_num_correct += (1 if static_correct == True else 0)
        static_total += label.size(0)

        if i % print_after == 0:
            tqdm.write(f"Optimal alpha accuracy:{num_correct / total}\tCorrect/total={num_correct}/{total}")
            tqdm.write(f"Static logit ensemble accuracy:{static_num_correct / static_total}\tCorrect/total={static_num_correct}/{static_total}")
            print_after *= 2
        i+=1 # Can't use enumerate, it messes up tqdm
    # print(alpha,loss,correct)  # Diagnostic - are these reasonable?
    print(f"Final Accuracy:{num_correct / total}, Correct:{num_correct}, Total: {total}")
    t2 = time.time()
    timeForOptimalAlphaAndStaticLogitEnsemble = t2-t0
    results['optimalAlpha'] = {"Accuracy":num_correct / total,"Time_with_static_logit_ensemble":timeForOptimalAlphaAndStaticLogitEnsemble}
    model1_acc,model1_time = get_top1_acc(model1,dataset_name,args.batch_size,args)
    results['model1'] = {"Accuracy":model1_acc,"Time":model1_time}
    model2_acc,model2_time = get_top1_acc(model1,dataset_name,args.batch_size,args)
    results['model2'] = {"Accuracy":model2_acc,"Time":model2_time}
    
    t0 = time.time()
    static_wse_model = getStaticWSE(model1,model2)
    static_wse_model_acc = get_top1_acc(static_wse_model,dataset_name,args.batch_size,args,returnTime=False)
    t1 = time.time()
    static_wse_model_time = t1-t0
    results['static_wse'] = {"Accuracy":static_wse_model_acc,"Time":static_wse_model_time}
    results['static_logit_ensemble'] = {"Accuracy":static_num_correct / static_total,"Time_with_optimalAlpha": timeForOptimalAlphaAndStaticLogitEnsemble}

    print(results)
    with open(args.results_db,'w') as f:
        f.write(json.dumps(results))
    return results

def getDataloader(dataset_name,preprocess_fn,data_location,batch_size=1,is_train=False,**kwargs):
    dataset_class = getattr(datasets,dataset_name)
    dataset = dataset_class(
        preprocess_fn,
        location=data_location,
        batch_size=batch_size
    )
    data_loader = dataset.train_loader if is_train else dataset.test_loader
    return dataset, data_loader

def useLogitEnsemble(label,loss_fn,logits1,logits2,dataset,args,alpha):
    ensembled_logits = alpha*logits1 + (1-alpha)*logits2
    ensemble_loss = loss_fn(ensembled_logits,label)
    pred = ensembled_logits.argmax(dim=1).to(args.device)
    correct = pred.eq(label.view_as(pred)).sum().item()  # Likely to cause problems I think
    total = label.size(0)
    # print(f"\rpred,label = {pred.item(),label.view_as(pred).item()}\t correct/total={correct}/{total}")
    # time.sleep(0.1)
    correct = (correct == total)  # Convert to boolean
    return ensemble_loss, correct

def getOptimalAlpha(logits1,logits2,label,loss_fn,args):
    alphas_to_try = torch.tensor([0.01*i for i in range(101)])
    one_minus_alphas_to_try = torch.tensor([1-0.01*i for i in range(101)])
    labels = torch.tensor([label]*101).to(args.device)
    alphas_to_try = torch.unsqueeze(alphas_to_try, 1)
    one_minus_alphas_to_try = torch.unsqueeze(one_minus_alphas_to_try, 1)
    # print(alphas_to_try.shape,one_minus_alphas_to_try.shape)  # torch.Size([101,1])
    alphas_to_try,one_minus_alphas_to_try = alphas_to_try.to(args.device),one_minus_alphas_to_try.to(args.device)
    logit_ensembles = torch.mul(alphas_to_try,logits1) + torch.mul(one_minus_alphas_to_try,logits2)
    losses = loss_fn(logit_ensembles,labels)
    optimalAlpha = 0.01*torch.argmin(losses)
    return optimalAlpha

def get_top1_acc(model,dataset_name,batch_size,args,returnTime=True):
    startTime = time.time()
    dataset,data_loader = getDataloader(dataset_name,model.val_preprocess,args.data_location,batch_size)
    num_correct, total = 0,0
    for batch in tqdm(data_loader):
        images,labels = batch['images'],batch['labels']
        if hasattr(dataset, 'project_labels'):
            labels = dataset.project_labels(labels, args.device)
        images,labels,model = images.to(args.device),labels.to(args.device),model.to(args.device)
        logits = model(images)
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, args.device)
        pred = logits.argmax(dim=1).to(args.device)
        num_correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)
    top_1_acc = num_correct / total
    endTime = time.time()
    totalTime = endTime - startTime
    if returnTime:
        return top_1_acc, totalTime
    else:
        return top_1_acc

def getStaticWSE(model1,model2,alpha=0.5):
    theta_0 = {k: v.clone() for k, v in model1.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in model2.state_dict().items()}
    assert set(theta_0.keys()) == set(theta_1.keys())
    theta = {key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()}
    static_wse_model = copy.deepcopy(model1)
    static_wse_model.load_state_dict(theta)
    return static_wse_model

def getStaticLogitEnsemble_top1_acc(model1,model2,dataset_name,batch_size,args):
    pass

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
    print(f"parsed_args.device={parsed_args.device}")
    return parsed_args

def makePlots(resultsNumbersList,resultsFolderPath="/shared/share_mala/rohan/test_and_open_clip/wise-ft/optimal_alpha_results"):
    num_to_dataset_name_dict = {1:"ImageNet",2:"ImageNetV2",3:"ImageNetR",4:"ImageNetSketch",5:"ImageNetA",6:"ObjectNet"}
    assert 1 in resultsNumbersList, "1 should be in resultsNumbersList, as it contains the ID (ImageNet) accuracy."
    model_accuracies = {}
    for result_num in resultsNumbersList:
        dataset_name = num_to_dataset_name_dict[result_num]
        results_db = f"results{result_num}.jsonl"
        with open(os.path.join(resultsFolderPath,results_db),"r") as f:
            results = json.loads(f.read())
        for key in results.keys():
            if key not in model_accuracies.keys():
                model_accuracies[key] = {dataset_name:get_acc(results[key])}  # Create inner dict
            else:
                model_accuracies[key][dataset_name] = get_acc(results[key])   # Add to inner dict
    for key in model_accuracies.keys():
        ood_accs = [model_accuracies[key][dataset_name] for dataset_name in model_accuracies[key].keys() if dataset_name is not "ImageNet"]
        mean_ood_acc = np.mean(ood_accs)
        model_accuracies[key]['Mean_OOD'] = mean_ood_acc
    first_model_accs_dict = model_accuracies[list(model_accuracies.keys())[0]]
    for dataset_name in first_model_accs_dict.keys():
        if dataset_name is "ImageNet":
            pass
        else:
            for model in model_accuracies.keys():
                id_acc = model_accuracies[model]["ImageNet"]
                ood_acc = model_accuracies[model][dataset_name]
                plt.plot(id_acc,ood_acc,'o',markersize=5,label=model)
            title = f"ImageNet_vs_{dataset_name}_Accuracy"
            plt.title(title)
            plt.xlabel("ImageNet Accuracy")
            plt.ylabel(f"{dataset_name} Accuracy")
            plt.legend()
            plot_path = os.path.join(resultsFolderPath,"plots",title)
            try:
                plt.savefig(plot_path,bbox_inches='tight')
            except FileNotFoundError:
                os.mkdir(plot_path[:-(len(title)+1)])
                plt.savefig(plot_path,bbox_inches='tight')
            plt.clf()
            print(f"{title} plot completed and saved to {plot_path}.")

def get_acc(dict_or_float):
    if isinstance(dict_or_float,float):
        return dict_or_float
    else:
        return dict_or_float["Accuracy"]

if __name__ == "__main__":
    # args = parse_arguments3()
    # model1,model2,dataset_name = ImageClassifier.load(args.model_ckpts[0]),ImageClassifier.load(args.model_ckpts[1]),args.dataset
    # sizeOpportunity(model1,model2,dataset_name,args)

    makePlots(resultsNumbersList=[1,2,3,4,5,6])