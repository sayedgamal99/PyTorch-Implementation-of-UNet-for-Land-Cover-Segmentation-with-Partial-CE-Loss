import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime

from .metrics import MetricsTracker
from .losses import PartialCrossEntropyLoss
from .utils import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer=None,
        scheduler=None,
        device='cuda',
        mode='partial_ce',
        num_classes=5,
        ignore_index=-1,
        save_dir='runs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_acc': []
        }

        self.best_miou = 0.0
        self.current_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch_idx, batch in enumerate(pbar):
            images, masks = batch
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return avg_loss

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        metrics_tracker = MetricsTracker(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index
        )

        pbar = tqdm(self.val_loader, desc='Validation')

        with torch.no_grad():
            for batch in pbar:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                total_loss += loss.item()
                num_batches += 1

                predictions = logits.argmax(dim=1)
                metrics_tracker.update(predictions, masks)

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics = metrics_tracker.get_metrics()

        return avg_loss, metrics

    def fit(self, num_epochs, save_best=True, save_every=None):
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.val_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_metrics['mean_iou'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}")

            if save_best and val_metrics['mean_iou'] > self.best_miou:
                self.best_miou = val_metrics['mean_iou']
                best_path = self.save_dir / 'best_model.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics, best_path
                )
                print(f"  ✓ New best mIoU: {self.best_miou:.4f}")

            if save_every and (epoch + 1) % save_every == 0:
                checkpoint_path = self.save_dir / \
                    f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics, checkpoint_path
                )

        print("\nTraining completed!")
        print(f"Best validation mIoU: {self.best_miou:.4f}")

        return self.history
