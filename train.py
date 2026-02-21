import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import TUNEPlusPlus, TUNELoss
from data import get_dataloader
from utils.metrics import compute_dice, compute_ece


def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    epoch_loss = 0
    epoch_losses = {'loss_seg': 0, 'loss_topo': 0, 'loss_unc': 0, 'loss_calib': 0, 'loss_hier': 0}
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long().squeeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss, loss_dict = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        for k in epoch_losses.keys():
            epoch_losses[k] += loss_dict[k]
        
        pbar.set_postfix({'total': loss.item()})
    
    for k in epoch_losses.keys():
        epoch_losses[k] /= len(loader)
    
    return epoch_loss / len(loader), epoch_losses


def validate(model, loader, device, num_classes):
    """Validation"""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device).long().squeeze(1)
            
            outputs = model(images)
            pred = outputs['segmentation']
            
            dice = compute_dice(pred, labels, num_classes)
            dice_scores.append(dice)
    
    dice_scores = np.array(dice_scores)
    return dice_scores.mean(axis=0)


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = TUNEPlusPlus(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss
    criterion = TUNELoss(
        lambda1=config['loss']['lambda1'],
        lambda2=config['loss']['lambda2'],
        lambda3=config['loss']['lambda3'],
        lambda4=config['loss']['lambda4'],
        w_b=config['loss']['topology_weights']['w_b'],
        w_j=config['loss']['topology_weights']['w_j'],
        w_a=config['loss']['topology_weights']['w_a']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Data loaders
    print("Loading datasets...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Logging
    import os
    os.makedirs('runs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    writer = SummaryWriter(f"runs/{config['dataset']['name']}")
    
    best_dice = 0
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        train_loss, train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_dice = validate(model, val_loader, device, config['dataset']['num_classes'])
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        for k, v in train_losses.items():
            writer.add_scalar(f'Loss/{k}', v, epoch)
        writer.add_scalar('Dice/val_mean', val_dice.mean(), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice (mean): {val_dice.mean():.4f}")
        print(f"Val Dice (per class): {val_dice}")
        
        # Save best model
        if val_dice.mean() > best_dice:
            best_dice = val_dice.mean()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, 'saved_models/best_model.pth')
            print(f"✓ Saved best model with Dice: {best_dice:.4f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_dice.mean(),
            }, f'saved_models/checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TUNE++ model')
    parser.add_argument('--config', type=str, default='configs/synapse.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)