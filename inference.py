import torch
import yaml
import argparse
import numpy as np

from models import TUNEPlusPlus
from utils.uncertainty import monte_carlo_inference
from utils.metrics import compute_dice, compute_ece, compute_taus


def inference(config_path, checkpoint_path, input_image):
    """Run inference with uncertainty quantification"""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = TUNEPlusPlus(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads']
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Load input
    # TODO: Implement actual image loading
    # input_tensor = load_medical_image(input_image)
    # For now, placeholder
    input_tensor = torch.randn(1, 1, 96, 96, 96).to(device)
    
    # MC Dropout inference
    mc_samples = config['inference']['mc_samples']
    
    print(f"Running {mc_samples} MC samples...")
    mean_pred, aleatoric, epistemic = monte_carlo_inference(
        model, input_tensor, num_samples=mc_samples
    )
    
    total_uncertainty = aleatoric + epistemic
    
    # Get final prediction
    final_pred = mean_pred.argmax(dim=1)
    
    print("Inference complete")
    print(f"Prediction shape: {final_pred.shape}")
    print(f"Mean aleatoric uncertainty: {aleatoric.mean():.4f}")
    print(f"Mean epistemic uncertainty: {epistemic.mean():.4f}")
    print(f"Mean total uncertainty: {total_uncertainty.mean():.4f}")
    
    # TODO: Save outputs
    # save_nifti(final_pred, 'output_segmentation.nii.gz')
    # save_nifti(total_uncertainty, 'output_uncertainty.nii.gz')
    
    return final_pred, total_uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    
    inference(args.config, args.checkpoint, args.input)