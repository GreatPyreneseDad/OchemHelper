"""Unit tests for Molecular VAE model."""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.insert(0, '/Users/chris/ochem-helper/src')

from models.generative.smiles_vae import MolecularVAE, SMILESTokenizer


class TestSMILESTokenizer(unittest.TestCase):
    """Test cases for SMILES tokenizer."""
    
    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = SMILESTokenizer()
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        self.assertIsInstance(self.tokenizer.vocab, dict)
        self.assertIn('<PAD>', self.tokenizer.vocab)
        self.assertIn('<SOS>', self.tokenizer.vocab)
        self.assertIn('<EOS>', self.tokenizer.vocab)
        self.assertIn('<UNK>', self.tokenizer.vocab)
        
        # Check special tokens have correct indices
        self.assertEqual(self.tokenizer.pad_idx, 0)
        self.assertEqual(self.tokenizer.sos_idx, 1)
        self.assertEqual(self.tokenizer.eos_idx, 2)
        self.assertEqual(self.tokenizer.unk_idx, 3)
    
    def test_encode_basic(self):
        """Test basic SMILES encoding."""
        smiles = "CCO"
        encoded = self.tokenizer.encode(smiles)
        
        self.assertIsInstance(encoded, list)
        self.assertEqual(encoded[0], self.tokenizer.sos_idx)
        self.assertEqual(encoded[-1], self.tokenizer.eos_idx)
        self.assertGreater(len(encoded), 2)  # At least SOS, EOS, and some tokens
    
    def test_encode_complex(self):
        """Test complex SMILES encoding."""
        smiles = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
        encoded = self.tokenizer.encode(smiles)
        
        self.assertIsInstance(encoded, list)
        self.assertEqual(encoded[0], self.tokenizer.sos_idx)
        self.assertEqual(encoded[-1], self.tokenizer.eos_idx)
        
        # Should handle brackets and special characters
        self.assertIn(self.tokenizer.vocab.get('(', self.tokenizer.unk_idx), encoded)
        self.assertIn(self.tokenizer.vocab.get(')', self.tokenizer.unk_idx), encoded)
        self.assertIn(self.tokenizer.vocab.get('=', self.tokenizer.unk_idx), encoded)
    
    def test_decode_basic(self):
        """Test basic decoding."""
        # Encode then decode
        smiles = "CCO"
        encoded = self.tokenizer.encode(smiles)
        decoded = self.tokenizer.decode(encoded)
        
        self.assertEqual(decoded, smiles)
    
    def test_decode_strips_special_tokens(self):
        """Test that decode strips special tokens."""
        # Manual token sequence with padding
        tokens = [self.tokenizer.sos_idx, 
                 self.tokenizer.vocab.get('C', self.tokenizer.unk_idx),
                 self.tokenizer.vocab.get('C', self.tokenizer.unk_idx),
                 self.tokenizer.eos_idx,
                 self.tokenizer.pad_idx,
                 self.tokenizer.pad_idx]
        
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "CC")
    
    def test_pad_sequences(self):
        """Test sequence padding."""
        sequences = [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2]
        ]
        
        padded = self.tokenizer.pad_sequences(sequences, max_length=6)
        
        self.assertEqual(padded.shape, (3, 6))
        self.assertEqual(padded[0].tolist(), [1, 2, 3, 0, 0, 0])
        self.assertEqual(padded[1].tolist(), [1, 2, 3, 4, 5, 0])
        self.assertEqual(padded[2].tolist(), [1, 2, 0, 0, 0, 0])
    
    def test_pad_sequences_truncate(self):
        """Test sequence padding with truncation."""
        sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8]
        ]
        
        padded = self.tokenizer.pad_sequences(sequences, max_length=5)
        
        self.assertEqual(padded.shape, (1, 5))
        self.assertEqual(padded[0].tolist(), [1, 2, 3, 4, 5])


class TestMolecularVAE(unittest.TestCase):
    """Test cases for Molecular VAE model."""
    
    def setUp(self):
        """Set up test VAE model."""
        self.vocab_size = 50
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.latent_dim = 32
        self.num_layers = 2
        self.max_length = 100
        
        self.vae = MolecularVAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            max_length=self.max_length
        )
    
    def test_initialization(self):
        """Test VAE initialization."""
        self.assertIsInstance(self.vae, MolecularVAE)
        self.assertEqual(self.vae.vocab_size, self.vocab_size)
        self.assertEqual(self.vae.latent_dim, self.latent_dim)
        
        # Check components exist
        self.assertIsNotNone(self.vae.embedding)
        self.assertIsNotNone(self.vae.encoder)
        self.assertIsNotNone(self.vae.decoder)
        self.assertIsNotNone(self.vae.fc_mu)
        self.assertIsNotNone(self.vae.fc_logvar)
        self.assertIsNotNone(self.vae.fc_latent)
        self.assertIsNotNone(self.vae.output_projection)
    
    def test_encode(self):
        """Test encoding functionality."""
        batch_size = 4
        seq_length = 20
        
        # Create dummy input
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        
        mu, logvar = self.vae.encode(x)
        
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        batch_size = 4
        
        mu = torch.randn(batch_size, self.latent_dim)
        logvar = torch.randn(batch_size, self.latent_dim)
        
        z = self.vae.reparameterize(mu, logvar)
        
        self.assertEqual(z.shape, (batch_size, self.latent_dim))
        
        # Test that it's stochastic
        z1 = self.vae.reparameterize(mu, logvar)
        z2 = self.vae.reparameterize(mu, logvar)
        self.assertFalse(torch.allclose(z1, z2))
    
    def test_decode(self):
        """Test decoding functionality."""
        batch_size = 4
        
        z = torch.randn(batch_size, self.latent_dim)
        
        output = self.vae.decode(z)
        
        self.assertEqual(output.shape, (batch_size, self.max_length, self.vocab_size))
    
    def test_forward(self):
        """Test full forward pass."""
        batch_size = 4
        seq_length = 20
        
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        
        recon_x, mu, logvar = self.vae(x)
        
        self.assertEqual(recon_x.shape, (batch_size, self.max_length, self.vocab_size))
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
    
    def test_loss_function(self):
        """Test loss computation."""
        batch_size = 4
        seq_length = 20
        
        # Create dummy data
        recon_x = torch.randn(batch_size, self.max_length, self.vocab_size)
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        mu = torch.randn(batch_size, self.latent_dim)
        logvar = torch.randn(batch_size, self.latent_dim)
        
        # Pad x to match recon_x length
        x_padded = torch.nn.functional.pad(x, (0, self.max_length - seq_length))
        
        loss, recon_loss, kl_loss = self.vae.loss_function(recon_x, x_padded, mu, logvar)
        
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kl_loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    @patch('torch.randn')
    def test_generate(self, mock_randn):
        """Test molecule generation."""
        n_samples = 2
        device = 'cpu'
        
        # Mock latent samples
        mock_randn.return_value = torch.zeros(n_samples, self.latent_dim)
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "CCO"
        
        with patch.object(self.vae, 'tokenizer', mock_tokenizer):
            molecules = self.vae.generate(n_samples, device)
        
        self.assertEqual(len(molecules), n_samples)
        self.assertIsInstance(molecules, list)
        self.assertTrue(all(isinstance(mol, str) for mol in molecules))
    
    def test_sample_latent(self):
        """Test latent space sampling."""
        n_samples = 5
        
        z = self.vae.sample_latent(n_samples)
        
        self.assertEqual(z.shape, (n_samples, self.latent_dim))
        
        # Should be approximately standard normal
        self.assertAlmostEqual(z.mean().item(), 0.0, places=1)
        self.assertAlmostEqual(z.std().item(), 1.0, places=1)
    
    def test_interpolate(self):
        """Test latent space interpolation."""
        batch_size = 2
        seq_length = 20
        n_steps = 5
        
        # Create two dummy molecules
        mol1 = torch.randint(0, self.vocab_size, (1, seq_length))
        mol2 = torch.randint(0, self.vocab_size, (1, seq_length))
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "CCO"
        
        with patch.object(self.vae, 'tokenizer', mock_tokenizer):
            interpolated = self.vae.interpolate(mol1, mol2, n_steps)
        
        self.assertEqual(len(interpolated), n_steps)
        self.assertIsInstance(interpolated, list)
        self.assertTrue(all(isinstance(mol, str) for mol in interpolated))
    
    def test_encode_molecules(self):
        """Test encoding of SMILES strings."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.pad_sequences.return_value = torch.randint(
            0, self.vocab_size, (len(molecules), 10)
        )
        
        with patch.object(self.vae, 'tokenizer', mock_tokenizer):
            mu, logvar = self.vae.encode_molecules(molecules)
        
        self.assertEqual(mu.shape, (len(molecules), self.latent_dim))
        self.assertEqual(logvar.shape, (len(molecules), self.latent_dim))
    
    def test_reconstruct(self):
        """Test molecule reconstruction."""
        molecules = ["CCO", "CC(=O)O"]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.pad_sequences.return_value = torch.randint(
            0, self.vocab_size, (len(molecules), 10)
        )
        mock_tokenizer.decode.return_value = "CCO"
        
        with patch.object(self.vae, 'tokenizer', mock_tokenizer):
            reconstructed = self.vae.reconstruct(molecules)
        
        self.assertEqual(len(reconstructed), len(molecules))
        self.assertIsInstance(reconstructed, list)
        self.assertTrue(all(isinstance(mol, str) for mol in reconstructed))


class TestTrainingUtils(unittest.TestCase):
    """Test training utilities and functions."""
    
    def test_model_training_mode(self):
        """Test model can be set to training mode."""
        vae = MolecularVAE(vocab_size=50)
        
        vae.train()
        self.assertTrue(vae.training)
        
        vae.eval()
        self.assertFalse(vae.training)
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        vae = MolecularVAE(vocab_size=50)
        
        params = list(vae.parameters())
        self.assertGreater(len(params), 0)
        
        # Check all parameters require gradients
        self.assertTrue(all(p.requires_grad for p in params))
    
    def test_model_device_transfer(self):
        """Test model can be transferred to different devices."""
        vae = MolecularVAE(vocab_size=50)
        
        # Should work without error (even if no GPU available)
        vae.to('cpu')
        
        # Check parameters are on correct device
        for p in vae.parameters():
            self.assertEqual(p.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()