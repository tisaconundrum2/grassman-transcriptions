import torch
import torchaudio
import torchaudio.transforms as T

from src.interfaces import IAudioProcessor
from src.models.config import GrassmannConfig


class AudioService(IAudioProcessor):
    """Handles audio loading and mel spectrogram extraction."""

    def __init__(self, config: GrassmannConfig):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )

    def load(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        # Log mel
        mel = torch.log(mel.clamp(min=1e-9))
        return mel.squeeze(0)  # (n_mels, time)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean()
        std = features.std().clamp(min=1e-9)
        return (features - mean) / std
