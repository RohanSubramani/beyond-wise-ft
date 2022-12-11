declare -i runNum
# runNum=1

# datasets=(ImageNet ImageNetV2 ImageNetSketch)

# for dataset in ${datasets[@]}
# do
#     results="optimal_alpha_results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"

#     CUDA_VISIBLE_DEVICES=0,1
#     python stacking_opportunity_sizing.py   \
#         --batch-size=64  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --dataset=$dataset
#     wait
#     runNum+=1
# done

# declare -i runNum
# runNum=1

# datasets=(ImageNet ImageNetV2 ImageNetR ImageNetSketch ImageNetA ObjectNet)

# for dataset in ${datasets[@]}
# do
#     results="optimal_alpha_results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"

#     CUDA_VISIBLE_DEVICES=0,1
#     python stacking_opportunity_sizing.py   \
#         --batch-size=64  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --dataset=$dataset
#     wait
#     runNum+=1
# done

# declare -i runNum
# runNum=3

# datasets=(ImageNetR ImageNetA ObjectNet)  # Replacing some results, since optimalAlpha was incorrectly computed before projecting logits

# for dataset in ${datasets[@]}
# do
#     results="optimal_alpha_results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"

#     CUDA_VISIBLE_DEVICES=0,1
#     python stacking_opportunity_sizing.py   \
#         --batch-size=64  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --dataset=$dataset
#     wait
#     if [ $runNum == 3 ]
#     then
#         runNum=5
#     elif [ $runNum == 5 ]
#     then
#         runNum=6
#     fi
# done

# runNum=7

# datasets=(ImageNet ImageNetV2 ImageNetR ImageNetSketch ImageNetA ObjectNet)

# for dataset in ${datasets[@]}
# do
#     results="optimal_alpha_results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"

#     CUDA_VISIBLE_DEVICES=0,1
#     python stacking_opportunity_sizing.py   \
#         --batch-size=64  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --dataset=$dataset
#     wait
#     runNum+=1
# done

# 7-12 are different datasets, same two models (zeroshot and finetuned ViTB32_8), writes optimal alphas for all datasets to a file.

# runNum=13

# datasets=(ImageNetV2)

# for dataset in ${datasets[@]}
# do
#     results="optimal_alpha_results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"

#     CUDA_VISIBLE_DEVICES=0,1
#     python stacking_opportunity_sizing.py   \
#         --batch-size=64  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --dataset=$dataset
#     wait
#     runNum+=1
# done

# Description of 13 is below along with descriptions of 14-18

runNum=14

datasets=(ObjectNet DeterministicImageNet ImageNetR ImageNetSketch ImageNetA)

for dataset in ${datasets[@]}
do
    results="optimal_alpha_results/results"
    jsonl=".jsonl"
    results_db="$results$runNum$jsonl"
    echo results_db is $results_db

    pathStart="./models/wiseft/ViTB32_8/"
    model1="zeroshot.pt"
    model2="finetuned/checkpoint_10.pt"

    CUDA_VISIBLE_DEVICES=0,1
    python stacking_opportunity_sizing.py   \
        --batch-size=64  \
        --model_ckpts="$pathStart$model1","$pathStart$model2" \
        --results-db=$results_db  \
        --data-location=/shared/share_mala/data \
        --dataset=$dataset
    wait
    runNum+=1
done

# 13-18 are the same as 7-12 [different datasets, same two models (zeroshot and finetuned ViTB32_8), writes optimal alphas for all datasets to a file] but the alpha writing actually worked (hopefully). Also, it says DeterministicImageNet instead of ImageNet, but they behave the same (it just changes the key in the dictionary in the file where the alphas are saved). The order is also changed to have ImageNetV2 and ObjectNet first, so that the first datasets finish more quickly as a check that it is working.