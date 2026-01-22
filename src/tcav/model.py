import torch
import torchaudio
import pytorch_lightning as pl


class MusicGenreClassifier(pl.LightningModule):
    """Simple CNN-based music genre classifier."""
    
    def __init__(self, num_genres: int = 10):
        super().__init__()
        self.save_hyperparameters()
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64
        )
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Bottleneck layer (for TCAV)
        self.bottleneck = torch.nn.Linear(128, 256)
        self.classifier = torch.nn.Linear(256, num_genres)
        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, return_bottleneck: bool = True):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        if x.shape[1] == 1 and x.shape[2] > 1000:
            x = self.mel_spectrogram(x)
            x = torchaudio.transforms.AmplitudeToDB()(x)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        bottleneck = self.bottleneck(x)
        x = self.relu(bottleneck)
        logits = self.classifier(x)
        
        if return_bottleneck:
            return logits, bottleneck
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, return_bottleneck=False)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, return_bottleneck=False)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
