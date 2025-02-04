import argparse
from datasets import Synapse_dataset

try:
    from datasets import ACDC_dataset
except:
    pass

try:
    from datasets import UAV_HSI_Crop_dataset
except:
    pass


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Synapse", help="experiment_name"
    )
    parser.add_argument(
        "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
    )
    parser.add_argument(
        "--num_classes", type=int, default=9, help="output channel of network"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.01,
        help="segmentation network learning rate",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="input patch size of network input",
    )
    parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
    parser.add_argument(
        "--n_skip",
        type=int,
        default=3,
        help="using number of skip-connect, default is num",
    )
    parser.add_argument(
        "--vit_name", type=str, default="R50+ViT-B_16", help="select one vit model"
    )
    parser.add_argument(
        "--vit_patches_size",
        type=int,
        default=16,
        help="vit_patches_size, default is 16",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=300, help="maximum epoch number to train"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed, default is 1234"
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="/project/mhssain9",
        help="training location",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="num of workers loading data",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=3,
        help="num of transformers",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="location for loading pretrained weights",
    )
    parser.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="if want to freeze transformer weights",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="if want to fine tune weights",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        default="/project/mhssain9/data/Synapse/train_npz",
        help="root dir for data",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")

    parser.add_argument(
        "--volume_path",
        type=str,
        default="/project/mhssain9/data/Synapse/test_vol_h5",
        help="root dir for validation volume data",
    )  # for acdc volume_path=root_dir

    parser.add_argument(
        "--is_savenii",
        action="store_true",
        help="whether to save results during inference",
    )
    parser.add_argument(
        "--test_save_dir",
        type=str,
        default="/project/mhssain9/predictions",
        help="saving prediction as nii!",
    )

    return parser


def get_snapshot_path(args):
    snapshot_path = f"{args.snapshot_dir}/{args.exp}/TU"
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_" + str(args.max_iterations)[0:2] + "k"
        if args.max_iterations != 30000
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_epo_" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = (
        snapshot_path + "_lr_" + str(args.base_lr)
        if args.base_lr != 0.01
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )

    return snapshot_path

dataset_config = {
    "ACDC": {
        "Dataset": ACDC_dataset,
        "root_path": "/project/mhssain9/data/ACDC",
        "volume_path": "/project/mhssain9/data/ACDC",
        "list_dir": None,
        "num_classes": 4,
        "z_spacing": 5,
        "info": "3D",
    },
    "UAV_HSI_Crop": {
        "Dataset": UAV_HSI_Crop_dataset,
        "root_path": "/project/mhssain9/data/UAV-HSI-Crop-Dataset",
        "volume_path": "/project/mhssain9/data/UAV-HSI-Crop-Dataset",
        "list_dir": None,
        "num_classes": 30,
        "z_spacing": 5,
        "info": "hsi",
    },
    "Synapse": {
        "Dataset": Synapse_dataset,
        "root_path": "/project/mhssain9/data/Synapse/train_npz",
        # "root_path": r"D:\Downloads\project\project_TransUNet\data\Synapse\train_npz",
        "volume_path": "/project/mhssain9/data/Synapse/test_vol_h5",
        # "volume_path": r"D:\Downloads\project\project_TransUNet\data\Synapse\test_vol_h5",
        "list_dir": "./lists/lists_Synapse",
        "num_classes": 9,
        "z_spacing": 1,
    },
}
