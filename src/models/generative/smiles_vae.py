"""SMILES Variational Autoencoder for molecular generation.

This module implements a VAE that operates on SMILES strings, converting them
to one-hot encoded sequences for training molecular generation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

logger = logging.getLogger(__name__)


class SMILESTokenizer:
    """Tokenizer for SMILES strings with special tokens."""
    
    def __init__(self):
        """Initialize tokenizer with SMILES vocabulary."""
        # Common SMILES tokens
        self.tokens = [
            'PAD', 'BOS', 'EOS', 'UNK',  # Special tokens
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # Atoms
            'B', 'Si', 'Se', 'Te', 'As', 'Al', 'Zn', 'Ca', 'Mg',  # More atoms
            'c', 'n', 'o', 's', 'p',  # Aromatic atoms
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',  # Numbers
            '#', '=', '-', '(', ')', '[', ']', '+', '-',  # Bonds and charges
            '@', '@@', '/', '\\', '.', '%',  # Stereochemistry and misc
            'H', 'He', 'Li', 'Be', 'Na', 'K', 'Fe', 'Cu', 'Ni', 'Co',  # Additional elements
        ]
        
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.tokens)
        
        # Special token indices
        self.pad_idx = self.token_to_idx['PAD']
        self.bos_idx = self.token_to_idx['BOS']
        self.eos_idx = self.token_to_idx['EOS']
        self.unk_idx = self.token_to_idx['UNK']
    
    def tokenize(self, smiles: str) -> List[int]:
        """Convert SMILES string to token indices."""
        tokens = [self.bos_idx]
        
        i = 0
        while i < len(smiles):
            # Try two-character tokens first (Cl, Br, etc.)
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in self.token_to_idx:
                    tokens.append(self.token_to_idx[two_char])
                    i += 2
                    continue
            
            # Single character tokens
            char = smiles[i]
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            else:
                tokens.append(self.unk_idx)
            i += 1
        
        tokens.append(self.eos_idx)
        return tokens
    
    def detokenize(self, indices: List[int]) -> str:
        """Convert token indices back to SMILES string."""
        smiles = []
        for idx in indices:
            if idx == self.bos_idx or idx == self.eos_idx or idx == self.pad_idx:
                continue
            if idx == self.unk_idx:
                smiles.append('?')
            else:
                smiles.append(self.idx_to_token.get(idx, '?'))
        return ''.join(smiles)


class MolecularVAEEncoder(nn.Module):
    """Encoder network for molecular VAE."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode SMILES sequences to latent space.
        
        Args:
            x: Token indices [batch_size, seq_len]
            lengths: Actual sequence lengths [batch_size]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Embedding
        embedded = self.dropout(self.embedding(x))
        
        # Pack sequences for efficient LSTM processing
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM encoding
        _, (hidden, _) = self.lstm(packed)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Project to latent space
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar


class MolecularVAEDecoder(nn.Module):
    """Decoder network for molecular VAE."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_length: int = 100
    ):
        super().__init__()
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Project latent vector to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + latent_dim,  # Concatenate with latent at each step
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent vectors to SMILES sequences.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            target: Target sequences for teacher forcing [batch_size, seq_len]
            lengths: Target sequence lengths [batch_size]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden states from latent vector
        h_0 = self.latent_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.latent_to_cell(z).view(self.num_layers, batch_size, self.hidden_dim)
        
        if target is not None:
            # Teacher forcing during training
            max_length = target.size(1)
            embedded = self.dropout(self.embedding(target))
            
            # Expand latent vector to concatenate at each time step
            z_expanded = z.unsqueeze(1).expand(-1, max_length, -1)
            lstm_input = torch.cat([embedded, z_expanded], dim=-1)
            
            # LSTM decoding
            output, _ = self.lstm(lstm_input, (h_0, c_0))
            logits = self.output_projection(output)
            
        else:
            # Autoregressive generation during inference
            outputs = []
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            input_token[:, 0] = 1  # BOS token
            
            hidden = (h_0, c_0)
            
            for _ in range(self.max_length):
                embedded = self.embedding(input_token)
                z_expanded = z.unsqueeze(1)
                lstm_input = torch.cat([embedded, z_expanded], dim=-1)
                
                output, hidden = self.lstm(lstm_input, hidden)
                logits = self.output_projection(output)
                outputs.append(logits)
                
                # Sample next token
                input_token = logits.argmax(dim=-1)
                
                # Check for EOS token
                if (input_token == 2).all():  # EOS token index
                    break
            
            logits = torch.cat(outputs, dim=1)
        
        return logits


class MolecularVAE(nn.Module):
    """Complete VAE model for molecular generation."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        beta: float = 1.0  # KL divergence weight
    ):
        super().__init__()
        
        self.encoder = MolecularVAEEncoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, dropout
        )
        
        self.decoder = MolecularVAEDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, 
            num_layers, dropout, max_length
        )
        
        self.beta = beta
        self.tokenizer = SMILESTokenizer()
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Dictionary with reconstruction, mu, logvar, and latent vectors
        """
        # Encode
        mu, logvar = self.encoder(x, lengths)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode (shift target for teacher forcing)
        target = x[:, :-1]  # Remove last token
        reconstruction = self.decoder(z, target)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def loss(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with reconstruction and KL divergence.
        
        Args:
            output: Model output dictionary
            target: Target sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Dictionary with total loss and components
        """
        reconstruction = output['reconstruction']
        mu = output['mu']
        logvar = output['logvar']
        
        # Reconstruction loss (ignore padding)
        target = target[:, 1:]  # Remove BOS token
        reconstruction = reconstruction.reshape(-1, reconstruction.size(-1))
        target = target.reshape(-1)
        
        # Create mask for non-padding tokens
        mask = target != self.tokenizer.pad_idx
        
        recon_loss = F.cross_entropy(
            reconstruction[mask],
            target[mask],
            reduction='mean'
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Average over batch
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def generate(
        self,
        n_samples: int = 10,
        z: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ) -> List[str]:
        """
        Generate new molecules.
        
        Args:
            n_samples: Number of molecules to generate
            z: Optional latent vectors to decode
            device: Device to run generation on
            
        Returns:
            List of SMILES strings
        """
        self.eval()
        
        with torch.no_grad():
            if z is None:
                # Sample from prior
                z = torch.randn(n_samples, self.encoder.fc_mu.out_features).to(device)
            
            # Decode
            logits = self.decoder(z)
            
            # Convert to SMILES
            sequences = logits.argmax(dim=-1)
            smiles_list = []
            
            for seq in sequences:
                smiles = self.tokenizer.detokenize(seq.cpu().tolist())
                
                # Validate with RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Canonicalize SMILES
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                else:
                    # Try to fix common issues
                    smiles_fixed = self._fix_smiles(smiles)
                    if smiles_fixed:
                        smiles_list.append(smiles_fixed)
        
        return smiles_list
    
    def _fix_smiles(self, smiles: str) -> Optional[str]:
        """Attempt to fix invalid SMILES strings."""
        # Remove invalid characters
        smiles = ''.join(c for c in smiles if c not in '?')
        
        # Try to parse again
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        
        return None
    
    def encode_smiles(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode SMILES strings to latent vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Latent vectors [n_molecules, latent_dim]
        """
        self.eval()
        
        # Tokenize
        sequences = []
        lengths = []
        
        for smiles in smiles_list:
            tokens = self.tokenizer.tokenize(smiles)
            sequences.append(torch.tensor(tokens))
            lengths.append(len(tokens))
        
        # Pad sequences
        x = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_idx)
        lengths = torch.tensor(lengths)
        
        with torch.no_grad():
            mu, _ = self.encoder(x, lengths)
        
        return mu
    
    def interpolate(
        self,
        smiles1: str,
        smiles2: str,
        n_steps: int = 10
    ) -> List[str]:
        """
        Interpolate between two molecules in latent space.
        
        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES
            n_steps: Number of interpolation steps
            
        Returns:
            List of interpolated SMILES
        """
        # Encode both molecules
        z1 = self.encode_smiles([smiles1])
        z2 = self.encode_smiles([smiles2])
        
        # Linear interpolation in latent space
        interpolated = []
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            smiles_list = self.generate(1, z_interp)
            if smiles_list:
                interpolated.append(smiles_list[0])
        
        return interpolated
