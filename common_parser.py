import argparse


def get_common_parser(state="training"):
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
        "--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model"
    )
    parser.add_argument(
        "--vit_patches_size",
        type=int,
        default=16,
        help="vit_patches_size, default is 16",
    )    
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="maximum epoch number to train"
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

    if state == "training":
        parser.add_argument(
            "--root_path",
            type=str,
            default="/project/mhssain9/data/Synapse/train_npz",
            help="root dir for data",
        )
        parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")

    elif state == "test":
        parser.add_argument(
            "--volume_path",
            type=str,
            default="/project/mhssain9/data/Synapse/test_vol_h5",
            help="root dir for validation volume data",
        )  # for acdc volume_path=root_dir

        # parser.add_argument(
        #     "--num_classes", type=int, default=4, help="output channel of network"
        # )
        # parser.add_argument(
        #     "--max_iterations",
        #     type=int,
        #     default=20000,
        #     help="maximum epoch number to train",
        # )
        # parser.add_argument(
        #     "--max_epochs", type=int, default=30, help="maximum epoch number to train"
        # )
        parser.add_argument(
            "--is_savenii",
            action="store_true",
            help="whether to save results during inference",
        )
        parser.add_argument(
            "--test_save_dir",
            type=str,
            default="../predictions",
            help="saving prediction as nii!",
        )

    return parser
