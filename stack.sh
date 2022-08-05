runNum=36

base="models/wiseft/alphaModel_ViTB32_8_"
save="$base$runNum"
echo save is $save

results="results/results"
jsonl=".jsonl"
results_db="$results$runNum$jsonl"
echo results_db is $results_db

pathStart="./models/wiseft/ViTB32_8/"
model1="zeroshot.pt"
model2="finetuned/checkpoint_10.pt"

python stack.py   \
    --train-dataset=ImageNet  \
    --save=$save  \
    --epochs=10  \
    --lr=3e-3  \
    --data_augmentation=1  \
    --batch-size=64  \
    --cache-dir=cache  \
    --model=ViT-B-32-quickgelu  \
    --model_ckpts="$pathStart$model1","$pathStart$model2" \
    --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
    --results-db=$results_db  \
    --data-location=/shared/share_mala/data \
    --template=openai_imagenet_template  \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet \
    --freeze-encoder \
    # --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
wait