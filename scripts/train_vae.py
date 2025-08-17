"""Training script for Molecular VAE."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import argparse
import yaml
import json

from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer

logger = logging.getLogger(__name__)


class SMILESDataset(Dataset):
    """Dataset for SMILES strings."""
    
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        max_length: int = 100,
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            smiles_list: List of SMILES strings
            tokenizer: SMILES tokenizer
            max_length: Maximum sequence length
            augment: Whether to use SMILES augmentation
        """
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Filter valid SMILES
        self.valid_smiles = []
        for smiles in tqdm(smiles_list, desc="Validating SMILES"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Canonicalize
                canonical = Chem.MolToSmiles(mol)
                if len(canonical) <= max_length - 2:  # Account for BOS/EOS
                    self.valid_smiles.append(canonical)
        
        logger.info(f"Valid SMILES: {len(self.valid_smiles)}/{len(smiles_list)}")
    
    def __len__(self):
        return len(self.valid_smiles)
    
    def __getitem__(self, idx):
        smiles = self.valid_smiles[idx]
        
        # SMILES augmentation (randomize atom order)
        if self.augment and np.random.random() > 0.5:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Randomize atom order
                atom_order = list(range(mol.GetNumAtoms()))
                np.random.shuffle(atom_order)
                mol = Chem.RenumberAtoms(mol, atom_order)
                smiles = Chem.MolToSmiles(mol, canonical=False)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(smiles)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.eos_idx]
        
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for padding sequences."""
    lengths = torch.tensor([len(seq) for seq in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths


class VAETrainer:
    """Trainer for Molecular VAE."""
    
    def __init__(
        self,
        model: MolecularVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'models/checkpoints',
        use_wandb: bool = True
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, lengths) in enumerate(pbar):
            data = data.to(self.device)
            lengths = lengths.to(self.device)
            
            # Forward pass
            output = self.model(data, lengths)
            losses = self.model.loss(output, data, lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['loss'].item(),
                'recon': losses['recon_loss'].item(),
                'kl': losses['kl_loss'].item()
            })
        
        n_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_kl_loss': total_kl_loss / n_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        valid_molecules = 0
        total_molecules = 0
        
        with torch.no_grad():
            for data, lengths in tqdm(self.val_loader, desc="Validation"):
                data = data.to(self.device)
                lengths = lengths.to(self.device)
                
                # Forward pass
                output = self.model(data, lengths)
                losses = self.model.loss(output, data, lengths)
                
                # Update metrics
                total_loss += losses['loss'].item()
                total_recon_loss += losses['recon_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                
                # Check validity of reconstructions
                recon = output['reconstruction'].argmax(dim=-1)
                for seq in recon:
                    smiles = self.model.tokenizer.detokenize(seq.cpu().tolist())
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        valid_molecules += 1
                    total_molecules += 1
        
        n_batches = len(self.val_loader)
        validity = valid_molecules / max(total_molecules, 1)
        
        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
            'val_kl_loss': total_kl_loss / n_batches,
            'val_validity': validity
        }
    
    def sample_molecules(self, n_samples: int = 10) -> Dict[str, any]:
        """Sample and evaluate generated molecules."""
        molecules = self.model.generate(n_samples, device=self.device)
        
        # Calculate metrics
        valid_count = 0
        unique_molecules = set()
        properties = []
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                unique_molecules.add(smiles)
                
                # Calculate properties
                props = {
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Crippen.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'QED': Descriptors.qed(mol),
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'Rotatable': Descriptors.NumRotatableBonds(mol)
                }
                properties.append(props)
        
        validity = valid_count / n_samples
        uniqueness = len(unique_molecules) / max(valid_count, 1)
        
        # Average properties
        if properties:
            avg_properties = {
                key: np.mean([p[key] for p in properties])
                for key in properties[0].keys()
            }
        else:
            avg_properties = {}
        
        return {
            'validity': validity,
            'uniqueness': uniqueness,
            'n_unique': len(unique_molecules),
            'molecules': list(unique_molecules)[:5],  # Sample for logging
            'properties': avg_properties
        }
    
    def train(self, n_epochs: int, sample_every: int = 5):
        """Full training loop."""
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Sample molecules
            if epoch % sample_every == 0:
                sample_metrics = self.sample_molecules(100)
            else:
                sample_metrics = {}
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, **sample_metrics}
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(metrics, step=epoch)
            
            # Print metrics
            logger.info(f"Epoch {epoch}: {metrics}")
            
            # Learning rate scheduling
            if val_metrics:
                self.scheduler.step(val_metrics['val_loss'])
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, metrics, is_best=True)
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, metrics)
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'vocab_size': self.model.encoder.embedding.num_embeddings,
                'embedding_dim': self.model.encoder.embedding.embedding_dim,
                'hidden_dim': self.model.encoder.lstm.hidden_size,
                'latent_dim': self.model.encoder.fc_mu.out_features,
                'num_layers': self.model.encoder.lstm.num_layers,
                'max_length': self.model.decoder.max_length,
                'beta': self.model.beta
            }
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def load_data(data_path: str, max_molecules: Optional[int] = None) -> List[str]:
    """Load SMILES data from file."""
    file_path = Path(data_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Try common column names
        for col in ['SMILES', 'smiles', 'Smiles', 'structure']:
            if col in df.columns:
                smiles_list = df[col].tolist()
                break
        else:
            # Use first column
            smiles_list = df.iloc[:, 0].tolist()
    
    elif file_path.suffix == '.txt':
        with open(file_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                smiles_list = data
            else:
                smiles_list = data.get('molecules', [])
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    if max_molecules:
        smiles_list = smiles_list[:max_molecules]
    
    logger.info(f"Loaded {len(smiles_list)} SMILES from {data_path}")
    return smiles_list


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Molecular VAE')
    parser.add_argument('--data', type=str, required=True, help='Path to SMILES data')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='KL weight (beta-VAE)')
    parser.add_argument('--max-molecules', type=int, help='Max molecules to use')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints')
    parser.add_argument('--wandb-project', type=str, default='molecular-vae')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    # Load data
    smiles_list = load_data(args.data, args.max_molecules)
    
    # Create tokenizer
    tokenizer = SMILESTokenizer()
    
    # Split data
    n_val = int(len(smiles_list) * args.val_split)
    np.random.shuffle(smiles_list)
    train_smiles = smiles_list[n_val:]
    val_smiles = smiles_list[:n_val]
    
    # Create datasets
    train_dataset = SMILESDataset(train_smiles, tokenizer, augment=True)
    val_dataset = SMILESDataset(val_smiles, tokenizer, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    model = MolecularVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=2,
        dropout=0.2,
        max_length=100,
        beta=args.beta
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb
    )
    
    # Train
    trainer.train(args.epochs)
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.sample_molecules(1000)
    logger.info(f"Final metrics: {final_metrics}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
