import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from common_parser import get_common_parser, dataset_config, get_snapshot_path


def inference(args, model, test_save_path=None, data_type: str = "test"):
    if data_type == "val":
        db_test = args.Dataset(base_dir=args.root_path, split="val")
        testloader = DataLoader(
            db_test, batch_size=1, shuffle=False, num_workers=args.num_workers
        )

    else:
        db_test = args.Dataset(
            base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir
        )
        testloader = DataLoader(
            db_test, batch_size=1, shuffle=False, num_workers=args.num_workers
        )
    logging.info("%s test iterations per epoch", len(testloader))

    is_rgb = False
    if args.dataset == "UAV_HSI_Crop":
        is_rgb = True

    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[-2:]
        image, label, case_name = (
            sampled_batch["image"],
            sampled_batch["label"],
            sampled_batch["case_name"][0],
        )
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
            is_rgb=is_rgb,
        )
        metric_list += np.array(metric_i)
        logging.info(
            "idx %s case %s mean_dice %s mean_hd95 %s mean_jaccard %s",
            i_batch,
            case_name,
            np.mean(metric_i, axis=0)[0],
            np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2],
        )

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            "Mean class %d mean_dice %s mean_hd95 %s mean_jaccard %s",
            i,
            metric_list[i - 1][0],
            metric_list[i - 1][1],
            metric_list[i - 1][2],
        )
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    logging.info(
        "Testing performance in best val model: mean_dice : %s mean_hd95 : %s mean_jaccard : %s",
        performance,
        mean_hd95,
        mean_jaccard,
    )
    print("Testing Finished!")
    return None


def main():
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
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = "TU_" + dataset_name + "_" + str(args.img_size)

    snapshot_path = get_snapshot_path(args)
    snapshot_name = snapshot_path.split("/")[-1]

    log_folder = f"{args.snapshot_dir}/{args.exp}_test_log"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(snapshot_name)
    logging.info(str(args))

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.transformer.num_layers = args.num_transformer_layers
    config_vit.n_classes = args.num_classes

    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    config_vit.n_skip = 0
    if args.vit_name.find("R50") != -1:
        config_vit.n_skip = args.n_skip

        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ViT_seg(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).cuda()

    snapshot = os.path.join(snapshot_path, "best_model.pth")
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "epoch_" + str(args.max_epochs))
        if not os.path.exists(snapshot):
            snapshot = snapshot.replace(
                "epoch_" + str(args.max_epochs), "epoch_" + str(args.max_epochs - 1)
            )

    logging.info("Loading model weight: %s", snapshot)
    net.load_state_dict(torch.load(snapshot))

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.snapshot_dir, args.exp + "_predictions")
        test_save_path = os.path.join(args.test_save_dir, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    logging.info(
        "-> Running Evaluation on Validation dataset to ensure correct model picked."
    )
    inference(args, net, data_type="val")

    logging.info("-> Running Evaluation on Test dataset.")
    inference(args, net, test_save_path)


if __name__ == "__main__":
    main()
