import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import DiceLoss, test_single_volume


"""TensorBoardX lets you watch Tensors Flow in PyTorch without Tensorflow
https://github.com/lanpa/tensorboardX/
"""

def trainer_acdc(args, model, snapshot_path):
    from datasets import ACDC_dataset, RandomGenerator

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = ACDC_dataset(
        base_dir=args.root_path,
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
    )
    db_val = ACDC_dataset(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("%s iterations per epoch", len(train_loader))
    logging.info("%s val iterations per epoch", len(val_loader))
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f",
                iter_num,
                loss.item(),
                loss_ce.item(),
            )

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image("train/Image", image, iter_num)
                outputs = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True
                )
                writer.add_image("train/Prediction", outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_loader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(
                        image,
                        label,
                        model,
                        classes=num_classes,
                        patch_size=[args.img_size, args.img_size],
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = (
                        iter_num,
                        performance,
                        mean_hd95,
                    )
                    save_best = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(model.state_dict(), save_best)
                    logging.info(
                        "Best model | iteration %d : mean_dice : %f mean_hd95 : %f",
                        iter_num,
                        performance,
                        mean_hd95,
                    )

                logging.info(
                    "iteration %d : mean_dice : %f mean_hd95 : %f",
                    iter_num,
                    performance,
                    mean_hd95,
                )
                model.train()

            if iter_num >= max_iterations:
                break


def trainer_uav_hsi(args, model, snapshot_path):
    from datasets import UAV_HSI_Crop_dataset, RandomGenerator

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    db_train = UAV_HSI_Crop_dataset(
        base_dir=args.root_path,
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
    )
    db_val = UAV_HSI_Crop_dataset(base_dir=args.root_path, split="val")

    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.train()

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")

    def image_write_helper(
        image_batch: np.array,
        label_batch: np.array,
        predictions: np.array,
        local_num: int,
        prefix: str = "train",
    ) -> None:
        image = image_batch[1, ...]
        image = (image - image.min()) / (image.max() - image.min())
        writer.add_image(f"{prefix}/Image", image, local_num)

        labs = label_batch[1, ...].unsqueeze(0) * 50
        writer.add_image(f"{prefix}/GroundTruth", labs, local_num)

        predictions = torch.argmax(
            torch.softmax(predictions, dim=1), dim=1, keepdim=True
        )
        writer.add_image(f"{prefix}/Prediction", predictions[1, ...] * 50, local_num)

    logging.info("%s iterations per epoch", len(train_loader))
    logging.info("%s val iterations per epoch", len(val_loader))
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    iter_num = 0
    # max_epoch = max_iterations // len(train_loader) + 1
    max_epochs = args.max_epochs
    max_iterations = max_epochs * len(train_loader)

    best_performance = 0.0
    iterator = tqdm(range(1, max_epochs), ncols=70)
    for epoch_num in iterator:
        loss_list, loss_ce_list, loss_dice_list = [], [], []

        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.to(dev), label_batch.to(dev)
            outputs = model(volume_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            loss_list.append(float(loss.item()))
            loss_ce_list.append(float(loss_ce.item()))
            loss_dice_list.append(float(loss_dice.item()))

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f",
                iter_num,
                loss.item(),
                loss_ce.item(),
            )

            if iter_num % 20 == 0:
                image_write_helper(volume_batch, label_batch, outputs, iter_num)

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_

        writer.add_scalar("info/lr", lr_, epoch_num)
        writer.add_scalar("info/total_loss", np.asarray(loss_list).mean(), epoch_num)
        writer.add_scalar("info/loss_ce", np.asarray(loss_ce_list).mean(), epoch_num)
        writer.add_scalar(
            "info/loss_dice", np.asarray(loss_dice_list).mean(), epoch_num
        )

        if epoch_num % 20 == 0:
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(val_loader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                image, label = image.to(dev), label.to(dev)

                outputs =  model(image)
                image_write_helper(image, label, outputs, epoch_num, prefix="val")

                metric_i = test_single_volume(
                    image,
                    label,
                    model,
                    classes=num_classes,
                    patch_size=[args.img_size, args.img_size],
                )
                metric_list += np.array(metric_i)
            
            metric_list = metric_list / len(db_val)
            for class_i in range(num_classes - 1):
                writer.add_scalar(
                    f"info/val_{class_i + 1}_dice",
                    metric_list[class_i, 0],
                    iter_num,
                )
                writer.add_scalar(
                    f"info/val_{class_i + 1}_hd95",
                    metric_list[class_i, 1],
                    iter_num,
                )

            performance = np.mean(metric_list, axis=0)[0]

            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar("info/val_mean_dice", performance, iter_num)
            writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

            if performance > best_performance:
                best_performance = performance

                save_best = os.path.join(snapshot_path, "best_model.pth")
                logging.info("Saving Best model | iteration %d %s", iter_num, save_best)
                torch.save(model.state_dict(), save_best)

                logging.info(
                    "Best model | iteration %d : mean_dice : %f mean_hd95 : %f",
                    iter_num,
                    performance,
                    mean_hd95,
                )

            logging.info(
                "Cur model | iteration %d : mean_dice : %f mean_hd95 : %f",
                iter_num,
                performance,
                mean_hd95,
            )
            model.train()

        
        if epoch_num >= max_epochs - 1:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            logging.info("Saving Final model | epoch: %d %s", epoch_num, save_mode_path)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to %s", save_mode_path)
            

        if iter_num >= max_iterations:
            iterator.close()
            break

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    logging.info("Training Finished!")


def trainer_synapse(args, model, snapshot_path):
    from datasets import RandomGenerator, Synapse_dataset

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )
    print(f"The length of train set is: {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    writer = SummaryWriter(snapshot_path + "/log")
    iter_num = 0
    max_epoch = args.max_epochs
    # max_epoch = max_iterations // len(train_loader) + 1
    max_iterations = args.max_epochs * len(train_loader)
    logging.info(
        "%d iterations per epoch. %d max iterations", len(train_loader), max_iterations
    )

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f",
                iter_num,
                loss.item(),
                loss_ce.item(),
            )

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image("train/Image", image, iter_num)
                outputs = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True
                )
                writer.add_image("train/Prediction", outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to %s", save_mode_path)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to %s", save_mode_path)
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
