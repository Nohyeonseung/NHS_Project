import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

from utils import ensure_dir, setup_logging, save_checkpoint, load_checkpoint
from dataloader import load_data_paths, get_dataloaders
from model import UNet3D
from loss import DiceCELoss
from metric import dice_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        if torch.isnan(loss):
            logging.warning("NaN detected in loss during training; skipping this batch.")
            continue
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device, num_classes, ignore_index):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            if torch.isnan(loss):
                logging.warning("NaN detected in loss during validation; skipping this batch.")
                continue

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            dice = dice_score(preds, masks, num_classes, ignore_index)
            dice_scores.append(dice)

    avg_loss = running_loss / len(dataloader.dataset)
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return avg_loss, avg_dice

def train_model(train_loader, val_loader, model, optimizer, criterion, device, num_classes, num_epochs, checkpoint_dir, ignore_index):
    ensure_dir(checkpoint_dir)
    setup_logging(checkpoint_dir)

    start_epoch, best_dice = load_checkpoint(model, optimizer, checkpoint_dir)
    
    for epoch in range(start_epoch, num_epochs + 1):
        logging.info(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, num_classes, ignore_index)

        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if not np.isnan(val_dice) and val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_loss, best_dice, checkpoint_dir, is_best=True)
        else:
            save_checkpoint(model, optimizer, epoch, val_loss, best_dice, checkpoint_dir, is_best=False)

if __name__ == '__main__':
    IMAGE_DIR = '/workspace/eye2brain/data/img'
    MASK_DIR = '/workspace/eye2brain/data/lbl'
    CHECKPOINT_DIR = "checkpoints_unet3d"

    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    INPUT_CHANNELS = 1
    BASE_CHANNELS = 16
    IGNORE_INDEX = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, mask_paths = load_data_paths(IMAGE_DIR, MASK_DIR)
    dataloaders = get_dataloaders(image_paths, mask_paths, batch_size=BATCH_SIZE)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    model = UNet3D(in_channels=INPUT_CHANNELS, out_channels=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = DiceCELoss(ignore_index=IGNORE_INDEX)

    train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion,
        DEVICE,
        NUM_CLASSES,
        NUM_EPOCHS,
        CHECKPOINT_DIR,
        IGNORE_INDEX
    )
