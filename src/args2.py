import os
import argparse

import torch

def parse_arguments2():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('./data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
             " Note that same model used for all datasets, so must have same classnames"
             "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="Dataset to train alpha model on, after adding logits from models to be ensembled",
    )
    parser.add_argument(
        "--model_ckpts",
        default=None,
        type=lambda x: x.split(","),
        help='Ckpt files for models being ensembled. Split by comma, e.g. "$pathStart$model1","$pathStart$model2" if pathStart=./models/wiseft/ViTB32_20/, model1=checkpoint_1.pt, model2=checkpoint_10.pt',
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="The alpha model ckpt, if loading one.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--data_augmentation",
        type=int,
        default=1,
        help="A few options for data augmentation",
    )
    parser.add_argument(
        "--subset_proportion",
        "-sp",
        type=float,
        default=1.0,
        help="Proportion of train dataset to use."
    )
    parser.add_argument(
        "--diagnostic_test",
        default=False,
        action="store_true",
        help="If True, multiplies logits of first model by -1 (so alpha model should learn to use 2nd model)."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Which optimizer to use? Options: AdamW, SGD"
    )
    parser.add_argument(
        "--eval_num",
        type=int,
        default=0,
        help="Number of evaluations on val dataset(s) completed so far - probably should never be modified from 0, useful to have as an argument though. Used in eval.py.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
