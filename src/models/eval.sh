# This is intended to compute the accuracy with a 50-50 logit ensemble, but it doesn't work yet

declare -i runNum
runNum=1

base="models/wiseft/stack_"
og_run="ViTB32_8_"
train_dataset="DeterministicImageNet"
save="$base$og_run$train_dataset$runNum"
echo save is $save

results="results/results"
jsonl=".jsonl"
results_db="$results${runNum}_reg_logit_ensemble${jsonl}"
echo results_db is $results_db

pathStart="./models/wiseft/ViTB32_8/"
model1="zeroshot.pt"
model2="finetuned/checkpoint_10.pt"
    
python src/models/eval.py   \
    --train-dataset=$train_dataset  \
    --subset_proportion=0.01  \
    --save=$save  \
    --epochs=10  \
    --lr=3e-3  \
    --wd=$wd  \
    --data_augmentation=8  \
    --batch-size=64  \
    --cache-dir=cache  \
    --model=ViT-B-32-quickgelu  \
    --model_ckpts="$pathStart$model1","$pathStart$model2" \
    --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
    --results-db=$results_db  \
    --data-location=/shared/share_mala/data \
    --template=openai_imagenet_template  \
    --eval-datasets=DeterministicImageNet,ImageNetV2 \
    --freeze-encoder