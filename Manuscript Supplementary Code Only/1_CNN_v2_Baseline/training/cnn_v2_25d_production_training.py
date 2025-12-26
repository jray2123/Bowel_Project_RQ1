#!/usr/bin/env python3
"""
Enhanced 2.5D CNN Training Script

This script trains the CNN25DEnhanced model following the Enhanced 2.5D Baseline 
Execution Plan specifications for Phase 4: Training Execution.

Training Command (from execution plan):
```bash
cd /mnt/HDD4/jineel/bowel_project/Baseline

# Run augmentation check first
python scripts/augmentation_sanity_check.py --config experiments/bowel_25d_cnn_enhanced/config.json

# Start training in tmux
tmux new-session -d -s cnn_25d_enhanced
tmux send-keys -t cnn_25d_enhanced "python scripts/train_25d_cnn_enhanced.py --config experiments/bowel_25d_cnn_enhanced/config.json" Enter
```
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset_25d_enhanced import create_dataloaders_25d
from models.cnn_25d_enhanced import create_cnn_25d_enhanced
from training.trainer_25d_enhanced import create_enhanced_25d_trainer


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced 2.5D CNN')
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--device', default='auto', help='Training device')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check augmentation validation
    config_path = Path(args.config)
    experiment_dir = config_path.parent
    aug_check_file = experiment_dir / 'augmentation_check_passed.txt'
    
    if not aug_check_file.exists():
        print("❌ ERROR: Augmentation sanity check not passed!")
        print("Run: python scripts/augmentation_sanity_check.py --config", args.config)
        sys.exit(1)
    
    print("✅ Augmentation sanity check passed")
    
    # Create data loaders
    print("Creating data loaders...")
    data_config = config['data']
    
    train_loader, val_loader = create_dataloaders_25d(
        data_dir=data_config['data_dir'],
        labels_csv=data_config['labels_csv'],
        train_patients_csv=data_config['train_patients_csv'],
        val_patients_csv=data_config['val_patients_csv'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        target_slice_size=data_config['target_slice_size'],
        max_sequence_length=data_config['max_sequence_length'],
        sequence_sampling=data_config['sequence_sampling'],
        position_encoding=data_config['position_encoding'],
        enable_augmentation=data_config['enable_augmentation'],
        debug_mode=False,
        test_mode=data_config.get('test_mode', False),
        max_train_patients=data_config.get('max_train_patients', None),
        max_val_patients=data_config.get('max_val_patients', None)
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating CNN25DEnhanced model...")
    model = create_cnn_25d_enhanced(config)
    
    model_info = model.get_model_info()
    print(f"  Model: {model_info['model_name']}")
    print(f"  Parameters: {model_info['total_parameters']:,}")
    print(f"  Memory estimate: {model_info['memory_estimate_mb']:.1f} MB")
    
    # Create trainer
    print("Creating enhanced trainer...")
    trainer = create_enhanced_25d_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=str(experiment_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_bowel_sensitivity = checkpoint['best_bowel_sensitivity']
        trainer.training_history = checkpoint['training_history']
        
    # Start training
    print("Starting training...")
    training_history = trainer.train()
    
    print("Training completed!")
    print(f"Best bowel sensitivity: {trainer.best_bowel_sensitivity:.4f}")
    print(f"Total epochs: {len(training_history)}")
    
    # Save final results
    results = {
        'config': config,
        'training_history': training_history,
        'best_bowel_sensitivity': trainer.best_bowel_sensitivity,
        'model_info': model_info
    }
    
    results_path = experiment_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()