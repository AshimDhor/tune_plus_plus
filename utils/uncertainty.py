import torch
import torch.nn.functional as F


def monte_carlo_inference(model, input_tensor, num_samples=25):
    """
    Perform Monte Carlo Dropout inference
    
    Args:
        model: Neural network with dropout
        input_tensor: Input image
        num_samples: Number of MC samples
    Returns:
        mean_pred: Mean prediction
        aleatoric: Aleatoric uncertainty
        epistemic: Epistemic uncertainty
    """
    model.train()  # Enable dropout
    
    predictions = []
    aleatorics = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(input_tensor)
            predictions.append(F.softmax(outputs['segmentation'], dim=1))
            aleatorics.append(outputs['aleatoric_uncertainty'])
    
    predictions = torch.stack(predictions)
    aleatorics = torch.stack(aleatorics)
    
    # Mean prediction
    mean_pred = predictions.mean(dim=0)
    
    # Epistemic uncertainty (predictive variance)
    epistemic = predictions.var(dim=0).sum(dim=1, keepdim=True)
    
    # Aleatoric uncertainty (mean of predicted variance)
    aleatoric = aleatorics.mean(dim=0)
    
    return mean_pred, aleatoric, epistemic


def compute_uncertainty_metrics(uncertainty, predictions, targets):
    """
    Compute uncertainty quality metrics
    
    Args:
        uncertainty: Total uncertainty map
        predictions: Model predictions
        targets: Ground truth
    Returns:
        metrics: Dictionary of uncertainty metrics
    """
    # Compute correctness
    correct = (predictions.argmax(dim=1) == targets).float()
    
    # Correlation between uncertainty and error
    correlation = torch.corrcoef(torch.stack([
        uncertainty.flatten(),
        (1 - correct).flatten()
    ]))[0, 1]
    
    metrics = {
        'uncertainty_error_correlation': correlation.item(),
        'mean_uncertainty': uncertainty.mean().item(),
        'std_uncertainty': uncertainty.std().item()
    }
    
    return metrics