import torch
import yaml
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import TUNEPlusPlus, TUNELoss
from data import get_dataloader
from utils.metrics import compute_dice, compute_ece


def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    epoch_loss = 0
    
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
        pbar.set_postfix(loss_dict)
    
    return epoch_loss / len(loader)


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
    
    return np.mean(dice_scores, axis=0)


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = TUNEPlusPlus(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads']
    ).to(device)
    
    # Loss
    criterion = TUNELoss(
        lambda1=config['loss']['lambda1'],
        lambda2=config['loss']['lambda2'],
        lambda3=config['loss']['lambda3'],
        lambda4=config['loss']['lambda4']
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
    
    # Data
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Logging
    writer = SummaryWriter(f"runs/{config['dataset']['name']}")
    
    best_dice = 0
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_dice = validate(model, val_loader, device, config['dataset']['num_classes'])
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/val_mean', val_dice.mean(), epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_dice.mean():.4f}")
        
        # Save best model
        if val_dice.mean() > best_dice:
            best_dice = val_dice.mean()
            # Create directory if it doesn't exist
            import os
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f"Saved best model with Dice: {best_dice:.4f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/synapse.yaml')
    args = parser.parse_args()
    
    main(args.config)