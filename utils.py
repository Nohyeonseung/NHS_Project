import os
import torch
import logging

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_logging(checkpoint_dir):
    ensure_dir(checkpoint_dir)
    log_file = os.path.join(checkpoint_dir, 'train_log.txt')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, epoch, val_loss, best_dice, checkpoint_dir, is_best):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_dice': best_dice
    }
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_checkpoint_path)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved at epoch {epoch} with Dice: {best_dice:.4f}")

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        logging.info(f"No checkpoint found. Starting training from scratch.")
        return 1, 0.0

    try:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        
        logging.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}.")
        return start_epoch, best_dice
        
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
        return 1, 0.0
