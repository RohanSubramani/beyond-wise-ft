declare -i runNum
runNum=4

datasets=(ImageNetSketch ImageNetA ObjectNet)

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