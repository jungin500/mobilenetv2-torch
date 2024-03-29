from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

import os

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torchvision.datasets import ImageNet

from model.mobilenet_v2 import MobileNetV2
from loguru import logger

import torch
from torch import optim
from torch import nn

import numpy as np
from torchinfo import summary

import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


class LightningModel(pl.LightningModule):
    def __init__(self,
                 base_path,
                 num_workers,
                 batch_size,
                 learning_rate,
                 enable_dali=False,
                 is_headless=False
                 ):
        super().__init__()

        self.model = MobileNetV2(num_classes=1000, width_mult=1.0)

        summary(self.model, (1, 3, 224, 224), device='cpu')

        # logger.info("Initializing weight (Kaiming for Conv2d, Xavier for Linear)")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.lr = learning_rate
        self.weight_decay = 1e-5
        self.momentum = 0.1
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.forward_idx = 0

        # if bool(config.model.pretrained.enabled):
        #     logger.info("Loading pretrained model")
        #     state_dict = torch.load(config.model.pretrained.path)
        #     self.load_state_dict(state_dict['state_dict'])
        # else:
        # logger.warning("Training from scratch without pretrained model")

        self.use_dali = enable_dali
        self.headless = is_headless
        self.dataset_path = base_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.m_acc = Accuracy()
        self.m_f1 = F1Score()
        self.m_precision = Precision()
        self.m_recall = Recall()

    def prepare_data(self, *args, **kwargs):
        if self.use_dali:
            return
        return super().prepare_data(*args, **kwargs)

    def setup(self, *args, **kwargs):
        if not self.use_dali:
            return super().setup(*args, **kwargs)

        # Start initilize DALI dataloader
        import nvidia.dali as dali
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types

        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
        import torch
        import os

        @pipeline_def
        def GetImageNetPipeline(device, data_path, shuffle, shard_id=0, num_shards=1):
            jpegs, labels = fn.readers.file(
                file_root=data_path,
                # random_shuffle=False,  # (shuffles inside a initial_fill)
                shuffle_after_epoch=shuffle,  # (shuffles entire datasets)
                name="Reader",
                shard_id=shard_id, num_shards=num_shards
            )
            images = fn.decoders.image(jpegs,
                                       device='mixed' if device == 'gpu' else 'cpu',
                                       output_type=types.DALIImageType.RGB)

            images = fn.resize(images, size=[224, 224])  # HWC
            images = fn.crop_mirror_normalize(images,
                                              dtype=types.FLOAT,
                                              scale=1 / 255.,
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225],
                                              output_layout="CHW")  # CHW
            if device == "gpu":
                labels = labels.gpu()
            # PyTorch expects labels as INT64
            labels = fn.cast(labels, dtype=types.INT64)

            return images, labels

        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        trainset_pipeline = GetImageNetPipeline(
            data_path=os.path.join(self.dataset_path, 'train'),
            batch_size=self.batch_size, device='gpu',
            shuffle=True,
            device_id=device_id, shard_id=shard_id,
            num_shards=num_shards, num_threads=self.num_workers
        )

        validset_pipeline = GetImageNetPipeline(
            data_path=os.path.join(self.dataset_path, 'val'),
            shuffle=False, device='gpu',
            device_id=device_id, shard_id=shard_id,
            # lot number of threads require lot of GPU memory
            batch_size=64, num_threads=2,
            num_shards=num_shards
        )

        logger.info("Creating train_loader")
        self.train_loader = LightningWrapper(
            trainset_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        logger.info("Creating valid_loader")
        self.valid_loader = LightningWrapper(
            validset_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    def train_dataloader(self, *args, **kwargs):
        if not self.use_dali:
            return super().train_dataloader(*args, **kwargs)
        return self.train_loader

    def val_dataloader(self, *args, **kwargs):
        if not self.use_dali:
            return super().val_dataloader(*args, **kwargs)
        return self.valid_loader

    def forward(self, images):
        print("Batch %05d -> Batches: %d" %
              (self.forward_idx, images.shape[0]))
        self.forward_idx += 1
        return self.model(images)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(
        ), lr=self.lr, weight_decay=self.weight_decay)  # , momentum=self.momentum)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.5, patience=2, verbose=False, threshold=3e-3, threshold_mode='abs')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "validation/accuracy"
        }

    def on_fit_start(self):
        if self.local_rank == 0 and self.headless:
            logger.info("Begin training in headless mode")

    def on_train_epoch_start(self) -> None:
        if not self.trainer.sanity_checking and self.headless:
            logger.info(
                f"[State={self.trainer.state.status}] Epoch {self.trainer.current_epoch} begin")
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.headless:
            logger.info(
                f"[State={self.trainer.state.status}] Epoch {self.trainer.current_epoch} end")
        return super().on_train_epoch_end()

    def training_epoch_end(self, outputs):
        self.forward_idx += 1
        return super().training_epoch_end(outputs)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        loss = self.loss_module(output, labels)

        with torch.no_grad():
            output = torch.argmax(torch.log_softmax(output, -1), -1)
            accuracy = (output == labels).float().mean()

            labels = labels.cpu().numpy()
            output = output.cpu().numpy()

        self.log("train/accuracy", accuracy, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        if self.local_rank == 0 and self.headless:
            if batch_idx % 10 == 0:
                print("\r\n", end='', flush=True)  # for Kubernetes to display progresbar correctly
        return super().on_train_batch_end(outputs, batch, batch_idx, unused)

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        output = self.model(images)
        output = torch.argmax(torch.log_softmax(output, -1), -1)
        
        self.m_acc.update(output, labels)
        self.m_precision.update(output, labels)
        self.m_recall.update(output, labels)
        self.m_f1.update(output, labels)
        
    def on_validation_epoch_start(self):        
        self.m_acc.reset()
        self.m_precision.reset()
        self.m_recall.reset()
        self.m_f1.reset()

    def on_validation_epoch_end(self):
        if '_update_count' in dir(self.m_acc) and self.m_acc._update_count == 0:
            logger.info("Skipped logging validation metrics")
            return
        
        accuracy = self.m_acc.compute()
        precision = self.m_precision.compute()
        recall = self.m_recall.compute()
        f1 = self.m_f1.compute()

        self.log("validation/accuracy", accuracy, prog_bar=True)
        self.log("validation/precision", precision)
        self.log("validation/recall", recall)
        self.log("validation/f1", f1)
        
        # Do not update scheduler while initializing
        if not self.trainer.sanity_checking:
            # Manually update lr_scheduler
            scheduler = self.lr_schedulers()
            # Manual scheduler step
            logger.info("Scheduler - accuracy step %.4f%%" % accuracy)
            scheduler.step(accuracy)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-base-dir', default='./outputs')
    parser.add_argument('--disable-wandb', default=False, action='store_true')
    parser.add_argument('--wandb-login-key', default=None)
    parser.add_argument('--wandb-project', default='mobilenetv3')
    parser.add_argument('--expr-name', default='mbv3-b128-1gpu-scratch')
    parser.add_argument('--enable-dali', '-n', default=False, action='store_true')
    parser.add_argument('--dataset-path', default='/datasets/imagenet')
    parser.add_argument('--wds-train-url', default='/datasets/imagenet/train', help='Dataset path only for webdataset')
    parser.add_argument('--wds-val-url', default='/datasets/imagenet/val', help='Dataset path only for webdataset')
    parser.add_argument('--webdataset', default=False, action='store_true', help='Enable webdataset with given --{train|val}-dataset-path')
    parser.add_argument('--batch-size', '-b', type=int, default=128)
    parser.add_argument('--num-workers', '-w', type=int, default=16)
    parser.add_argument('--num-gpus', '-g', type=int, default=1)
    parser.add_argument('--train-strategy', default='none', choices=['none', 'ddp'])
    parser.add_argument('--train-precision', type=int, default=16, choices=[16, 32])
    parser.add_argument('--train-epochs', type=int, default=20)
    parser.add_argument('--train-limit-batches', type=float, default=1.0)
    parser.add_argument('--val-limit-batches', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--headless', default=False, action='store_true')
    parser.add_argument('--early-stop', default=False, action='store_true', help='Enables early stopping')
    parser.add_argument('--resume', default=None, type=str, help='Resume after given checkpoint')
    parser.add_argument('--debug','-d', default=False, action='store_true', help="Debug mode (disables all automation including wandb")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.train_base_dir
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')

    # directory internally used by pytorch lightning
    trainer_root_dir = os.path.join(base_dir, 'trainer')
    os.makedirs(trainer_root_dir, exist_ok=True)

    wandb_enabled = not args.disable_wandb and not args.debug

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Created checkpoint directory {checkpoint_dir}")

    if wandb_enabled:
        # directory internally used by wandb
        wandb_log_dir = os.path.join(base_dir, 'wandb')
        os.makedirs(wandb_log_dir, exist_ok=True)

        if args.wandb_login_key != 'None' and args.wandb_login_key is not None:
            import wandb
            wandb.login(key=args.wandb_login_key)

        lightning_logger = WandbLogger(
            name=args.expr_name,
            project=args.wandb_project,
            save_dir=base_dir
        )
    else:
        lightning_logger = True

    if args.enable_dali:
        logger.info("Creating DALI pipeline inside LightningModule")
        pass  # Initializing inside model
    elif args.webdataset:
        import webdataset as wds
        from pathlib import Path
        from PIL import Image
        import io
        import cv2
        
        logger.info("Creating Webdataset dataloader")
        
        def cv2decode(value):
            image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            return image
        
        train_ds_url, val_ds_url = args.wds_train_url, args.wds_val_url
        train_dataset = wds.WebDataset(train_ds_url).shuffle(1000).decode(wds.handle_extension('jpg jpeg', cv2decode)).to_tuple("jpeg", "__key__")
        val_dataset = wds.WebDataset(val_ds_url).shuffle(1000).decode(wds.handle_extension('jpg jpeg', cv2decode)).to_tuple("jpeg", "__key__")
    
        synset_to_class_map = { k[0]: int(k[1]) for k in [line.split(' ') for line in open('imagenet_synset_to_class.txt', 'r').read().strip().splitlines()] }
        
        train_transform = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomAutocontrast(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        def create_preprocessor(transform=lambda image: image):
            def preprocess(sample):
                image, key = sample
                synset_name = str(Path(key).parent)
                class_id = synset_to_class_map[synset_name]
                
                image = transform(image)
                return image, class_id
            
            return preprocess
        
        train_dataset = train_dataset.map(create_preprocessor(train_transform))
        val_dataset = val_dataset.map(create_preprocessor(val_transform))
        
        train_loader = wds.WebLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        val_loader = wds.WebLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
    else:
        logger.info("Creating standard ImageNet dataloader")
        train_transform = T.Compose([
            T.Resize([224, 224]),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomAutocontrast(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("Loading trainset")
        train_dataset = ImageNet(
            root=args.dataset_path,
            split='train',
            transform=train_transform,
            # target_transform=target_transform
        )
        logger.info("Done loading trainset")

        logger.info("Loading validset")
        val_dataset = ImageNet(
            root=args.dataset_path,
            split='val',
            transform=val_transform,
            # target_transform=target_transform
        )
        logger.info("Done loading validset")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    model = LightningModel(
        args.dataset_path,
        args.num_workers,
        args.batch_size,
        learning_rate=args.learning_rate,
        enable_dali=args.enable_dali,
        is_headless=args.headless
    )

    if wandb_enabled and args.num_gpus <= 1:
        lightning_logger.watch(model)

    if args.num_gpus == 0:
        logger.warning("Training with CPU, falling back to BF16 format")

    strategy = None
    if args.train_strategy and args.train_strategy.lower() != 'none':
        strategy = args.train_strategy
        if strategy.lower() == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=False)

    if args.webdataset:
        args.train_limit_batches = int(1282000 / args.batch_size)
        args.val_limit_batches = int(50000 / args.batch_size)
        
    callbacks = [
        RichProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='%s-epoch{epoch:04d}-val_acc{validation/accuracy:.2f}' % (args.expr_name),
            auto_insert_metric_name=False,
            mode="max", monitor="validation/accuracy", verbose=True
        ),
        LearningRateMonitor("epoch")
    ]
    
    if args.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="validation/accuracy",
                patience=10,
                min_delta=0.005,
                mode="max",
                verbose=True
            )
        )
        
    trainer = pl.Trainer(
        logger=lightning_logger,
        default_root_dir=trainer_root_dir,
        accelerator='gpu' if args.num_gpus > 0 else 'cpu',
        gpus=args.num_gpus,
        strategy=strategy,
        precision=args.train_precision if args.num_gpus > 0 else 'bf16',
        max_epochs=args.train_epochs,
        limit_train_batches=args.train_limit_batches,
        limit_val_batches=args.val_limit_batches,
        callbacks=callbacks
    )

    if wandb_enabled:
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None

    if args.enable_dali:
        if args.resume:
            logger.info("Resuming from checkpoint %s ..." % (args.resume))
            trainer.fit(model, ckpt_path=args.resume)
        else:
            trainer.fit(model)  # Dataloader is initialized inside model
    else:
        if args.resume:
            logger.info("Resuming from checkpoint %s ..." % (args.resume))
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
        else:
            trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
