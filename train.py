import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse, trainer_acdc, trainer_uav_hsi
from common_parser import get_common_parser, dataset_config, get_snapshot_path


if __name__ == "__main__":
    parser = get_common_parser()
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.is_pretrain = True
    args.exp = "TU_" + dataset_name + "_" + str(args.img_size)

    snapshot_path = get_snapshot_path(args)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.transformer.num_layers = args.num_transformer_layers
    config_vit.n_classes = args.num_classes
    if args.checkpoint_path:
        config_vit.pretrained_path = args.checkpoint_path

    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    config_vit.n_skip = 0
    if args.vit_name.find("R50") != -1:
        config_vit.n_skip = args.n_skip

        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    # =========================================================================
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = VisionTransformer(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).to(dev)

    net.load_from(weights=np.load(config_vit.pretrained_path))
    if args.freeze_transformer:
        for name, p in net.named_parameters():
            if "encoder" in name:
                p.requires_grad = False

    trainer = {
        "Synapse": trainer_synapse,
        "ACDC": trainer_acdc,
        "UAV_HSI_Crop": trainer_uav_hsi,
    }
    trainer[dataset_name](args, net, snapshot_path, config_vit)
