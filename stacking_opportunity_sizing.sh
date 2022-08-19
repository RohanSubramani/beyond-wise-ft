python stacking_opportunity_sizing.py   \
    --batch-size=64  \
    --model_ckpts="$pathStart$model1","$pathStart$model2" \
    --results-db=$results_db  \
    --data-location=/shared/share_mala/data \
    --dataset=ImageNet
wait
    