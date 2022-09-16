declare -i runNum
# runNum=36
# base="models/wiseft/stack_"
# og_run="ViTB32_8_"
# train_dataset="DeterministicCIFAR10"
# save="$base$og_run$train_dataset_$runNum"
# echo save is $save

# results="results/results"
# jsonl=".jsonl"
# results_db="$results$runNum$jsonl"
# echo results_db is $results_db

# pathStart="./models/wiseft/ViTB32_8/"
# model1="zeroshot.pt"
# model2="finetuned/checkpoint_10.pt"

# python stack.py   \
#     --train-dataset=$train_dataset  \
#     --save=$save  \
#     --epochs=1  \
#     --lr=3e-3  \
#     --data_augmentation=1  \
#     --batch-size=64  \
#     --cache-dir=cache  \
#     --model=ViT-B-32-quickgelu  \
#     --model_ckpts="$pathStart$model1","$pathStart$model2" \
#     --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=CIFAR10,CIFAR101,CIFAR102 \
#     --freeze-encoder \
#     # --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# wait

# runNum=40
# base="models/wiseft/stack_"
# og_run="ViTB32_8_"
# train_dataset="DeterministicImageNet"
# save="$base$og_run${train_dataset}_$runNum"
# echo save is $save

# results="results/results"
# jsonl=".jsonl"
# results_db="$results$runNum$jsonl"
# echo results_db is $results_db

# pathStart="./models/wiseft/ViTB32_8/"
# model1="zeroshot.pt"
# model2="finetuned/checkpoint_10.pt"

# python stack.py   \
#     --train-dataset=$train_dataset  \
#     --subset_proportion=0.0001  \
#     --save=$save  \
#     --epochs=10  \
#     --lr=3e-3  \
#     --data_augmentation=1  \
#     --batch-size=64  \
#     --cache-dir=cache  \
#     --model=ViT-B-32-quickgelu  \
#     --model_ckpts="$pathStart$model1","$pathStart$model2" \
#     --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=DeterministicImageNet,ImageNetV2 \
#     --freeze-encoder \
#     # ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet
#     # --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     # --subset_proportion=0.001  \
# wait

# runNum=45

# lrs=(3e-3 3e-2 3e-1)

# # If all goes well, 45-47 will be 3 diagnostic test runs with different lr's, testing if linear probing with alpha model built on CLIP features can learn to select an actual model's predictions over the negated predictions of another model (which should be the opposite of good)

# for lr in ${lrs[@]}
# do
#     base="models/wiseft/stack_"
#     og_run="ViTB32_8_"
#     train_dataset="DeterministicImageNet"
#     save="$base$og_run$train_dataset$runNum"
#     echo save is $save

#     results="results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo results_db is $results_db

#     pathStart="./models/wiseft/ViTB32_8/"
#     model1="zeroshot.pt"
#     model2="finetuned/checkpoint_10.pt"
    
#     python stack.py   \
#         --train-dataset=$train_dataset  \
#         --subset_proportion=0.01  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=$lr  \
#         --data_augmentation=1  \
#         --batch-size=64  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=DeterministicImageNet,ImageNetV2 \
#         --freeze-encoder  \
#         --diagnostic_test
#     wait
#     runNum+=1
# done

runNum=48

lrs=(3e-3 3e-2 3e-1)

for lr in ${lrs[@]}
do
    base="models/wiseft/stack_"
    og_run="ViTB32_8_"
    train_dataset="DeterministicImageNet"
    save="$base$og_run$train_dataset$runNum"
    echo save is $save

    results="results/results"
    jsonl=".jsonl"
    results_db="$results$runNum$jsonl"
    echo results_db is $results_db

    pathStart="./models/wiseft/ViTB32_8/"
    model1="zeroshot.pt"
    model2="finetuned/checkpoint_10.pt"
    
    python stack.py   \
        --train-dataset=$train_dataset  \
        --subset_proportion=0.01  \
        --save=$save  \
        --epochs=10  \
        --lr=$lr  \
        --data_augmentation=1  \
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
    wait
    runNum+=1
done

# This ^ is 45-47 repeated without the diagnostic_test tag, meaning without sabotaging one of the model's by negating its logits