import enum
import matplotlib.pyplot as plt
import json
import os
import random
import math

def makePlots(results_db,eval_datasets,alphas,save,pathStart='/shared/share_mala/rohan/test_and_open_clip/wise-ft'):
    
    with open(os.path.join(pathStart,results_db),"r") as f:
        results = json.loads(f.read())
    
    ensemble_results = results[-(len(alphas)):] # Just from weight ensemble evaluations
    train_results = results[:-(len(alphas))]

    all_ys = {}
    all_early_stop_ood_accs = {}
    first = True
    for dataset in eval_datasets:
        xs = alphas
        ys = [result[dataset+":top1"] for result in ensemble_results]
        all_ys[dataset] = ys   # Dictionary of lists

        # else:
        #     early_stop_ood_accs = [result[dataset+":top1"] for result in train_results]
        #     all_early_stop_ood_accs[dataset] = early_stop_ood_accs

        # No plotting of early stopping when alphas is the x-axis, so just gathering relevant data here.

        plotInterpolation(xs,ys,alphas,label=dataset,first=first)
        
        first = False

        # plt.show()
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plotsFolder = pathStart+"/"+save[:-10]+"/plots" # Remove "/finetuned"
    if not os.path.isdir(plotsFolder):
        os.makedirs(plotsFolder)
    plt.savefig(plotsFolder+"/WiSE_FT_Plot",bbox_inches='tight')
    plt.clf()

    for j in range(2):
        if j==1:
            plotsFolder += "/plots_with_early_stopping"
            if not os.path.isdir(plotsFolder):
                os.makedirs(plotsFolder)
        for i,dataset in enumerate(eval_datasets):
            if i==0:
                target_dataset = dataset   # Usually ImageNet   # Assumes that the first el of eval_datasets is the target dataset
                id_accs=all_ys[dataset]  # In-distrubution accuracies

                early_stop_id_accs = [result[dataset+":top1"] for result in train_results]
            else:
                ood_dataset = dataset
                ood_accs = all_ys[dataset]
                early_stop_ood_accs = [result[dataset+":top1"] for result in train_results]
                all_early_stop_ood_accs[dataset] = early_stop_ood_accs
                
                if j == 1:
                    plotInterpolation(early_stop_id_accs,early_stop_ood_accs,alphas,withES=True,label='early stopping')
                plotInterpolation(id_accs,ood_accs,alphas)

                plt.xlabel(target_dataset + " Accuracy")
                plt.ylabel(ood_dataset + " Accuracy")
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.savefig(plotsFolder+f"/{target_dataset}_vs_{ood_dataset}",bbox_inches='tight')
                plt.clf()
        
        avg_ood_accs = [sum([all_ys[dataset][i] for dataset in eval_datasets[1:]])/(len(eval_datasets)-1) for i in range(len(alphas))]
        avg_es_ood_accs = [sum([all_early_stop_ood_accs[dataset][i] for dataset in eval_datasets[1:]])/(len(eval_datasets)-1) for i in 
                            range(len(train_results))]
        if j==1:
            plotInterpolation(early_stop_id_accs,avg_es_ood_accs,alphas,withES=True,label='early stopping')
        plotInterpolation(id_accs,avg_ood_accs,alphas)
        plt.xlabel(target_dataset + " Accuracy")
        plt.ylabel("Average OOD Accuracy")
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.savefig(plotsFolder+f"/{target_dataset}_vs_Average_OOD",bbox_inches='tight')
        plt.clf()

        print(f"Plots saved to {plotsFolder}.")


def plotInterpolation(xs,ys,alphas,label=None,first=True,withES=False):
    if withES:      # With accuracies with early stopping plotted
        if label is None:
            plt.plot(xs,ys,'ko--',markersize=2.5)
        else:
            plt.plot(xs,ys,'ko--',markersize=2.5,label=label)
    
    else:
        if label is None:
            plt.plot(xs,ys,'o-',markersize=3)
        else:
            plt.plot(xs,ys,'o-',markersize=3,label=label)

        if first:
            plt.plot(xs[0], ys[0], 'm*', markersize=9, label='zeroshot')
            mid = len(xs)//2
            plt.plot(xs[mid], ys[mid], 'cs', markersize=6, label=f"alpha={alphas[mid]}")
            plt.plot(xs[-1], ys[-1], 'gD', markersize=6, label=f"finetuned")
        else:
            plt.plot(xs[0], ys[0], 'm*', markersize=9)
            mid = len(xs)//2
            plt.plot(xs[mid], ys[mid], 'cs', markersize=6)
            plt.plot(xs[-1], ys[-1], 'gD', markersize=6)

def makeJointPlots(results_dbs,eval_datasets,alphas,save):  # save should be used instead of hardcoded paths
    
    maxRunNum = results_dbs[-1][15:-6]
    cube_side = math.ceil(len(results_dbs) ** (1. / 3)) # Used to iterate through colors
    color_grid = []
    for x in range(cube_side):
        for y in range(cube_side):
            for z in range(cube_side):
                r = 0.05 + 0.95*x/cube_side
                g = 0.05 + 0.95*y/cube_side
                b = 0.05 + 0.95*z/cube_side
                color_grid.append((r,g,b))
    
    details = {1:'lr=3e-7',2:'Basic (lr=3e-5)',3:'lr=3e-3',4:'lr=3e-4',5:'lr=No data aug',6:'H_Flip',7:'Rotation',8:'+H_Flip',9:'+H_Flip_and_Rotation',10:'ViT-L/14',11:'LP, lr=3e-5',12:'LP, lr=3e-4',13:'LP, lr=3e-3',14:'LP, lr=3e-2',15:'LP, lr=3e-1',16:'LP, lr=3e-2, +H_Flip',17:'LP, lr=3e-2, +H_Flip_and_Rotation',18:'+2 datasets, lr=3e-7',19:'+2D, lr=3e-6',20:'+2D, Basic (lr=3e-5)',21:'+2D, lr=3e-4',22:'+2D, +H_Flip',23:'+2D, +H_Flip_and_Rotation',24:'+2D, strong_aug_0.1',25:'+2D, strong_aug_0.3',26:'+2D, strong_aug_0.5',27:'+2D, strong_aug_0.7',28:'+2D, strong_aug_0.9'}
    
    for j in range(3):
        
        for k,results_db in enumerate(results_dbs):
            pathStart = '/shared/share_mala/rohan/test_and_open_clip/wise-ft' # FIXME This is too specific to my current file paths 
            with open(pathStart+'/'+results_db,"r") as f:
                results = json.loads(f.read())
            
            ensemble_results = results[-(len(alphas)):] # Just from weight ensemble evaluations
            train_results = results[:-(len(alphas))]

            # Get ID and average OOD accs for interpolation
            all_interpolation_ys = {dataset:[result[dataset+":top1"] for result in ensemble_results] for dataset in eval_datasets}
            interpolation_id_accs = all_interpolation_ys[eval_datasets[0]]
            avg_interpolation_ood_accs = [sum([all_interpolation_ys[dataset][i] for dataset in eval_datasets[1:]])/(len(eval_datasets)-1) 
                                            for i in range(len(alphas))]

            if j >= 1:
                plt.plot(interpolation_id_accs,avg_interpolation_ood_accs,'o-',markersize=3,color=color_grid[k])
            
            if j == 2:
                all_earlystop_ys = {dataset:[result[dataset+":top1"] for result in train_results] for dataset in eval_datasets}
                earlystop_id_accs = all_earlystop_ys[eval_datasets[0]]
                avg_earlystop_ood_accs = [sum([all_earlystop_ys[dataset][i] for dataset in eval_datasets[1:]]) / \
                                                (len(eval_datasets)-1) for i in range(len(train_results))]
                if results_db == results_dbs[0]:
                    plt.plot(earlystop_id_accs,avg_earlystop_ood_accs,'ko--',markersize=2.5,label='early stopping')
                else:
                    plt.plot(earlystop_id_accs,avg_earlystop_ood_accs,'ko--',markersize=2.5)
            

            plt.plot(interpolation_id_accs[0], avg_interpolation_ood_accs[0], '*', markersize=9)
            runNum = results_db[15:-6]
            keys = details.keys()
            if int(runNum) in keys:
                label = details[int(runNum)]
            else: 
                label = runNum
                # print(f"keys={keys}, does not contain {runNum}.")
            plt.plot(interpolation_id_accs[-1], avg_interpolation_ood_accs[-1], 'D', markersize=6, label=label,color=color_grid[k])
        plotsFolder = pathStart + '/models/wiseft/joint_plots'
        if not os.path.isdir(plotsFolder):
            os.makedirs(plotsFolder)
        
        plt.xlabel(eval_datasets[0] + " Accuracy")
        plt.ylabel("Average OOD Accuracy")
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        save_location = plotsFolder+f"/{eval_datasets[0]}_vs_Average_{len(eval_datasets)-1}_OOD_{maxRunNum}_{j}"
        plt.savefig(save_location,bbox_inches='tight')
        print(f"Plot saved to {save_location}")
        plt.clf()
    

if __name__ == "__main__":  # Just for a specific case I want to test
    # save = 'models/wiseft/ViTL14_10/finetuned'
    # makePlots('results/results10.jsonl',['ImageNet','ImageNetV2','ImageNetR','ImageNetSketch'],[0.0,0.1,0.3,0.5,0.7,0.9,1.0],save)

    results_dbs =  [f"results/results{i}.jsonl" for i in [1,2,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26]] 
    # Last number should be largest
    eval_datasets = ['ImageNet','ImageNetV2','ImageNetR','ImageNetSketch']
    alphas = [0.1*x for x in range(11)]
    makeJointPlots(results_dbs,eval_datasets,alphas,save=None)

    results_dbs =  [f"results/results{i}.jsonl" for i in [18,19,20,22,23,24,25,26]] 
    # Last number should be largest
    eval_datasets = ['ImageNet','ImageNetV2','ImageNetR','ImageNetSketch','ImageNetA','ObjectNet']
    alphas = [0.1*x for x in range(11)]
    makeJointPlots(results_dbs,eval_datasets,alphas,save=None)