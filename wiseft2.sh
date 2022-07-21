nvidia-smi

# export DATA_LOCATION=/shared/share_mala/data

# # CIFAR10.1
# mkdir -p $DATA_LOCATION/CIFAR-10.1
# wget https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy -P $DATA_LOCATION/CIFAR-10.1
# wget https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy -P $DATA_LOCATION/CIFAR-10.1

# # CIFAR10.2
# mkdir -p $DATA_LOCATION/CIFAR-10.2
# wget https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_test.npy -P $DATA_LOCATION/CIFAR-10.2
# wget https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_train.npy -P $DATA_LOCATION/CIFAR-10.2

# ImageNetA
# mkdir -p $DATA_LOCATION/ImageNetA
# cd $DATA_LOCATION/ImageNetA
# wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
# tar -xvf imagenet-a.tar
# rm imagenet-a.tar

# # ImageNet-R
# # mkdir -p $DATA_LOCATION/ImageNet-R
# wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
# tar -xvzf imagenet-r.tar
# rm imagenet-r.tar

# # ImageNet V2
# # mkdir -p $DATA_LOCATION/ImageNetV2
# wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
# tar -xvf imagenetv2-matched-frequency.tar.gz
# rm imagenetv2-matched-frequency.tar.gz

# ObjectNet
# mkdir -p $DATA_LOCATION/ObjectNet
# cd $DATA_LOCATION/ObjectNet
# wget https://objectnet.dev/downloads/objectnet-1.0.zip
# unzip objectnet-1.0.zip
# rm objectnet-1.0.zip

# cd src
# python plot.py

# declare -i runNum
# runNum=1

# lrs=(3e-7 3e-5 3e-3)   # Ran this before

# runNum=4

# lrs=(3e-4)
# for lr in ${lrs[@]}
# do
#     # String manipulation
#     echo lr is $lr
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo $save

#     results="results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     python wise_ft.py   \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=$lr  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     wait
#     runNum+=1
# done

# das=(2 3 4 5 6)  # data augmentations  # da=1 is runNum=2, when lr=3e-5.
# for da in ${das[@]}
# do
#     # String manipulation
#     echo da is $da
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo $save

#     results="results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     python wise_ft.py   \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=3e-5  \
#         --data_augmentation=$da  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     wait
#     runNum+=1
# done


# CUDA_VISIBLE_DEVICES=0,1,2,3 
# python wise_ft.py   \
#     --train-dataset=ImageNet  \
#     --save=models/wiseft/ViTB32_4  \
#     --epochs=10  \
#     --lr=0.00003  \
#     --batch-size=512  \
#     --cache-dir=cache  \
#     --model=ViT-B-32-quickgelu  \
#     --results-db=results.jsonl  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#     --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0  \

# --load ./models/wiseft/ViTB32_3/zeroshot.pt,./models/wiseft/ViTB32_3/finetuned/checkpoint_0.pt \
# ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet
# CIFAR10,CIFAR101,CIFAR102
# /shared/share_mala/data or ./data
# ViT-B/32 # RN50x16   <-- These were original options. New options:
# _PRETRAINED = {
#         "RN50": _RN50,
#         "RN50-quickgelu": _RN50_quickgelu,
#         "RN101": _RN101,
#         "RN101-quickgelu": _RN101_quickgelu,
#         "RN50x4": _RN50x4,
#         "RN50x16": _RN50x16,
#         "RN50x64": _RN50x64,
#         "ViT-B-32": _VITB32,
#         "ViT-B-32-quickgelu": _VITB32_quickgelu,
#         "ViT-B-16": _VITB16,
#         "ViT-B-16-plus-240": _VITB16_PLUS_240,
#         "ViT-L-14": _VITL14,
#         "ViT-L-14-336": _VITL14_336,
# }

# declare -i runNum
# runNum=10

# base="models/wiseft/ViTL14_"
# save="$base$runNum"
# echo $save

# results="results"
# jsonl=".jsonl"
# results_db="$results$runNum$jsonl"
# echo $results_db

# CUDA_VISIBLE_DEVICES=0,1,2,3 python wise_ft.py   \
#     --train-dataset=ImageNet  \
#     --save=$save  \
#     --epochs=10  \
#     --lr=3e-5  \
#     --data_augmentation=1  \
#     --batch-size=256  \
#     --cache-dir=cache  \
#     --model=ViT-L-14  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#     --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# wait
# runNum+=1

# base="models/wiseft/ViTB32_"
# save="$base$runNum"
# echo $save

# results="results"
# jsonl=".jsonl"
# results_db="$results$runNum$jsonl"
# echo $results_db

# CUDA_VISIBLE_DEVICES=0,1 #,2,3 
# python wise_ft.py   \
#     --train-dataset=ImageNet  \
#     --save=$save  \
#     --epochs=10  \
#     --lr=3e-3  \
#     --data_augmentation=1  \
#     --batch-size=512  \
#     --cache-dir=cache  \
#     --model=ViT-B-32  \
#     --results-db=$results_db  \
#     --data-location=/shared/share_mala/data \
#     --template=openai_imagenet_template  \
#     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#     --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
#     --freeze-encoder
# wait
# runNum+=1

# declare -i runNum
# runNum=11

# lrs=(3e-5 3e-4 3e-3 3e-2 3e-1)

# for lr in ${lrs[@]}
# do
#     # String manipulation
#     echo lr is $lr
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo $save

#     results="results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     CUDA_VISIBLE_DEVICES=0,1 python wise_ft.py  \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=$lr  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0  \
#         --freeze-encoder  # This is the new part - LP instead of E2E
#     wait
#     runNum+=1
# done

# declare -i runNum
# runNum=16

# das=(5 6)  # data augmentations  # da=1 is runNum=14, when lr=3e-2. 2,3,4 don't work
# for da in ${das[@]}
# do
#     # String manipulation
#     echo da is $da
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo $save

#     results="results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     CUDA_VISIBLE_DEVICES=0,1 python wise_ft.py   \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=3e-2  \
#         --data_augmentation=$da  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0  \
#         --freeze-encoder
#     wait
#     runNum+=1
# done

# declare -i runNum
# runNum=18

# lrs=(3e-7 3e-6 3e-5 3e-4)

# for lr in ${lrs[@]}
# do
#     # String manipulation
#     echo lr is $lr
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo save is $save

#     results="results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     python wise_ft.py   \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=$lr  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     wait
#     runNum+=1
# done

# das=(5 6)

# for da in ${das[@]}
# do
#     # String manipulation
#     echo da is $da
#     base="models/wiseft/ViTB32_"
#     save="$base$runNum"
#     echo save is $save

#     results="results/results"
#     jsonl=".jsonl"
#     results_db="$results$runNum$jsonl"
#     echo $results_db

#     python wise_ft.py   \
#         --train-dataset=ImageNet  \
#         --save=$save  \
#         --epochs=10  \
#         --lr=3e-5  \
#         --data_augmentation=$da  \
#         --batch-size=512  \
#         --cache-dir=cache  \
#         --model=ViT-B-32-quickgelu  \
#         --results-db=$results_db  \
#         --data-location=/shared/share_mala/data \
#         --template=openai_imagenet_template  \
#         --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet \
#         --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     wait
#     runNum+=1
# done

# ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet

declare -i runNum
runNum=32

das=(2 7 8 9)

for da in ${das[@]}
do
    # String manipulation
    echo da is $da
    base="models/wiseft/ViTB16_"
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
        --batch-size=64  \
        --cache-dir=cache  \
        --model=ViT-B-16  \
        --results-db=$results_db  \
        --data-location=/shared/share_mala/data \
        --template=openai_imagenet_template  \
        --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA,ObjectNet \
        --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    wait
    runNum+=1
done