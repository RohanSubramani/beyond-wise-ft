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

# runNum=48

# lrs=(3e-3 3e-2 3e-1)

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
#         --freeze-encoder
#     wait
#     runNum+=1
# done

# This ^ (48-50) is 45-47 repeated without the diagnostic_test tag, meaning without sabotaging one of the models by negating its logits.

# runNum=51

# # lr = 3e-3 was the best above, by a little bit
# das=(1 2 3 4 5 6 7 8 9 10 11) # Not sure if anything other than da=1 will work, have to think this through!!
# wds=(0.01 0.05 0.1 0.2 0.4)

# for da in ${das[@]}
# do
#     for wd in ${wds[@]}
#     do
#         base="models/wiseft/stack_"
#         og_run="ViTB32_8_"
#         train_dataset="DeterministicImageNet"
#         save="$base$og_run$train_dataset$runNum"
#         echo save is $save

#         results="results/results"
#         jsonl=".jsonl"
#         results_db="$results$runNum$jsonl"
#         echo results_db is $results_db

#         pathStart="./models/wiseft/ViTB32_8/"
#         model1="zeroshot.pt"
#         model2="finetuned/checkpoint_10.pt"
        
#         python stack.py   \
#             --train-dataset=$train_dataset  \
#             --subset_proportion=0.01  \
#             --save=$save  \
#             --epochs=10  \
#             --lr=3e-3  \
#             --wd=$wd  \
#             --data_augmentation=$da  \
#             --batch-size=64  \
#             --cache-dir=cache  \
#             --model=ViT-B-32-quickgelu  \
#             --model_ckpts="$pathStart$model1","$pathStart$model2" \
#             --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#             --results-db=$results_db  \
#             --data-location=/shared/share_mala/data \
#             --template=openai_imagenet_template  \
#             --eval-datasets=DeterministicImageNet,ImageNetV2 \
#             --freeze-encoder
#         wait
#         runNum+=1
#     done
# done

# ^ 51-105 was a nested for loop of data aug's and wd's. Best val acc on ImageNet was the largest wd, so should try larger ones.

# runNum=106

# # lr = 3e-3 was the best above, by a little bit
# wds=(0.7 1.5) # 1.0 2.0

# for wd in ${wds[@]}
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
#         --lr=3e-3  \
#         --wd=$wd  \
#         --data_augmentation=8  \
#         --batch-size=64  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=DeterministicImageNet,ImageNetV2 \
#         --freeze-encoder
#     wait
#     runNum+=1
# done

# runNum=110

# # lr = 3e-3 was the best above, by a little bit
# wds=(0.7 1.0 1.5 2.0) # 1.0 2.0

# for wd in ${wds[@]}
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
#         --lr=3e-3  \
#         --wd=$wd  \
#         --data_augmentation=8  \
#         --batch-size=64  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=DeterministicImageNet,ImageNetV2 \
#         --freeze-encoder
#     wait
#     runNum+=1
# done

# ^ 110 - 113 are the same as 106-109 (in a different order!), but now gradient norms and val loss are being tracked in wandb dashboard.

# runNum=114

# # lr = 3e-3 was the best above, by a little bit
# wds=(4.0 8.0 50.0 100.0 300.0) # 1.0 2.0

# for wd in ${wds[@]}
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
#         --lr=3e-3  \
#         --wd=$wd  \
#         --data_augmentation=8  \
#         --batch-size=64  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --model_ckpts="$pathStart$model1","$pathStart$model2" \
#         --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=DeterministicImageNet,ImageNetV2 \
#         --freeze-encoder
#     wait
#     runNum+=1
# done


# CUDA_VISIBLE_DEVICES=1

# runNum=122

# base="models/wiseft/stack_"
# og_run="ViTB32_8_"
# train_dataset="DeterministicImageNet"
# save="$base$og_run$train_dataset$runNum"
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
#     --subset_proportion=0.1  \
#     --save=$save  \
#     --epochs=10  \
#     --lr=3e-3  \
#     --wd=4.0  \
#     --data_augmentation=8  \
#     --batch-size=64  \
#     --cache-dir=cache  \
#     --model=ViT-B-32-quickgelu  \
#     --model_ckpts="$pathStart$model1","$pathStart$model2" \
#     --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=DeterministicImageNet,ImageNetV2 \
#     --optimizer="SGD"  \
#     --freeze-encoder
# wait

# 124 is the same as 120, except with subset proportion = 0.1 instead of 0.01.

# runNum=125

# wds=(4.0 8.0 50.0 100.0 300.0) # 1.0 2.0
# lrs=(3e-5 3e-4 3e-3 3e-2 3e-1)
# optimizers=("SGD" "AdamW")

# for wd in ${wds[@]}
# do
#     for lr in ${lrs[@]}
#     do
#         for optimizer in ${optimizers[@]}
#         do
#             base="models/wiseft/stack_"
#             og_run="ViTB32_8_"
#             train_dataset="DeterministicImageNet"
#             save="$base$og_run$train_dataset$runNum"
#             echo save is $save

#             results="results/results"
#             jsonl=".jsonl"
#             results_db="$results$runNum$jsonl"
#             echo results_db is $results_db

#             pathStart="./models/wiseft/ViTB32_8/"
#             model1="zeroshot.pt"
#             model2="finetuned/checkpoint_10.pt"

#             python stack.py   \
#                 --train-dataset=$train_dataset  \
#                 --subset_proportion=0.01  \
#                 --save=$save  \
#                 --epochs=7  \
#                 --lr=$lr  \
#                 --wd=$wd  \
#                 --data_augmentation=8  \
#                 --batch-size=64  \
#                 --cache-dir=cache  \
#                 --model=ViT-B-32-quickgelu  \
#                 --model_ckpts="$pathStart$model1","$pathStart$model2" \
#                 --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#                 --results-db=$results_db  \
#                 --data-location=/shared/share_mala/data \
#                 --template=openai_imagenet_template  \
#                 --eval-datasets=DeterministicImageNet,ImageNetV2 \
#                 --optimizer=$optimizer  \
#                 --freeze-encoder
#             wait
#             runNum+=1
#         done
#     done
# done
# ^ 125 is the first one-epoch test run following the attempted addition of saveComparisons function to stack.py.
# Data: .01*ImageNet. Objective: Minimize cross entropy loss of ensemble using predicted alpha. 
# Alpha model: "./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt". 
# Base models: Zeroshot and finetuned ViTB32_8. See hyperparams above.


# runNum=175

# base="models/wiseft/stack_"
# og_run="ViTB32_8_"
# train_dataset="DeterministicImageNet"
# save="$base$og_run$train_dataset$runNum"
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
#     --subset_proportion=1.0  \
#     --save=$save  \
#     --epochs=10  \
#     --lr=3e-1  \
#     --wd=8.0  \
#     --data_augmentation=8  \
#     --batch-size=64  \
#     --cache-dir=cache  \
#     --model=ViT-B-32-quickgelu  \
#     --model_ckpts="$pathStart$model1","$pathStart$model2" \
#     --load="./models/wiseft/ViTB32_8/finetuned/wise_ft_alpha=0.500.pt"  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=DeterministicImageNet,ImageNetV2 \
#     --optimizer="AdamW"  \
#     --freeze-encoder
# wait

# The run above was supposed to replicate a successful one from before, but it resulted in errors, including NaN alpha predictions. I'm not sure why yet.

runNum=176

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
    --subset_proportion=1.0  \
    --save=$save  \
    --epochs=10  \
    --lr=3e-1  \
    --wd=8.0  \
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
    --optimizer="AdamW"  \
    --freeze-encoder
wait

# 176 replicates 144, with subset proportion 1 and 10 epochs
# Which is an exact repeat of 175, which failed to replicate the results of 144 

runNum+=1

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
    --subset_proportion=1.0  \
    --save=$save  \
    --epochs=10  \
    --lr=3e-4  \
    --wd=4.0  \
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
    --optimizer="AdamW"  \
    --freeze-encoder
wait

# 177 replicates 128, with subset proportion 1 and 10 epochs

runNum+=1

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
    --subset_proportion=1.0  \
    --save=$save  \
    --epochs=10  \
    --lr=3e-3  \
    --wd=4.0  \
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
    --optimizer="AdamW"  \
    --freeze-encoder
wait

# 178 replicates 130, with subset proportion 1 and 10 epochs