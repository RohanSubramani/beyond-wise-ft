# declare -i runNum
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

declare -i runNum
runNum=3

datasets=(ImageNetR ImageNetA ObjectNet)  # Replacing some results, since optimalAlpha was incorrectly computed before projecting logits

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
    if [ $runNum == 3 ]
    then
        runNum=5
    elif [ $runNum == 5 ]
    then
        runNum=6
    fi
done