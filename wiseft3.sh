nvidia-smi

declare -i runNum
runNum=24

das=(7 8 9 10 11)

for da in ${das[@]}
do
    # String manipulation
    echo da is $da
    base="models/wiseft/ViTB32_"
    save="$base$runNum"
    echo save is $save

    results="results/results"
    jsonl=".jsonl"
    results_db="$results$runNum$jsonl"
    echo results_db is $results_db

    python wise_ft.py   \
        --train-dataset=ImageNet  \
        --save=$save  \
        --epochs=10  \
        --lr=3e-5  \
        --data_augmentation=$da  \
        --batch-size=512  \
        --cache-dir=cache  \
        --model=ViT-B-32-quickgelu  \
        --results-db=$results_db  \
        --data-location=/shared/share_mala/data \
        --template=openai_imagenet_template  \
        --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet \
        --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    wait
    runNum+=1
done