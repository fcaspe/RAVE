import torch
from typing import Optional,Literal

from typing import Optional,Literal
class SpectrogramMasking(torch.nn.Module):
    def __init__(self,
                 target_type: Literal['autocorrelation','relative_amp','relative_power'],
                 pow_times:int,
                 win_length:int,
                 hop_ratio:int,
                 mask_ratio: float = 0.125,
                 cutoff:Optional[int] = None):
        super().__init__()
        self.target_type = target_type
        self.pow_times = pow_times
        self.win_length = win_length
        self.hop_ratio = hop_ratio
        self.cutoff = cutoff
        self.mask_ratio = mask_ratio
        self.register_buffer('window',
            torch.hann_window(self.win_length)
        )

    @torch.no_grad()
    def forward(self,wave:torch.tensor):
        mag,phase = self.to_spectrogram(wave,
                                   self.window,
                                   hop_length=self.win_length//self.hop_ratio,                                
                                   win_length=self.win_length)
        wave_len = wave.shape[-1]
        if self.cutoff is not None:
            #c = self.cutoff/wave.sr
            c = self.cutoff/44100
            c = int(c*self.win_length//2)
            mag[:,c:,:] = 0
        mask = self.get_mask_spectrogram(mag,self.target_type)
        processed = mag*mask.pow(self.pow_times)
        reconstructed_signal_batch = self.to_signal(processed,
                                               phase,
                                               self.window,
                                               win_length=self.win_length,
                                               hop_length=self.win_length//self.hop_ratio,
                                               original_len=wave_len)
        return reconstructed_signal_batch

    def to_spectrogram(self, signal_batch, window, hop_length, win_length):
        """
        Perform STFT, convert to magnitude and phase, reconstruct complex spectrogram,
        and then perform ISTFT on a batch of signals.
        
        Parameters:
        signal_batch (torch.Tensor): Batch of input signals with shape (batch_size, signal_length).
        hop_length (int): Number of audio samples between adjacent STFT columns.
        win_length (int): Window size for STFT.
        
        Returns:
        reconstructed_signal_batch (torch.Tensor): Batch of reconstructed signals with shape (batch_size, signal_length).
        """
        # Ensure the input is a 2D tensor
        assert signal_batch.ndim == 2, "Input signal batch must be a 2D tensor with shape (batch_size, signal_length)"
        
        # Perform STFT
        stft_result = torch.stft(signal_batch, n_fft=win_length, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
        
        # Convert to magnitude and phase
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        
        return magnitude,phase

    def get_mask_spectrogram(self,spectrogram,
                             target_type: Literal['autocorrelation','relative_amp','relative_power','softmaxed_acorr']):

        mask = None
        if target_type == 'autocorrelation':
            # Normalize the spectrogram. Get time-wise maxes.
            mask_max_values = torch.max(spectrogram,dim=1)[0].unsqueeze(1)
            mask = spectrogram/mask_max_values

        elif target_type == 'relative_amp':
            # Normalize the spectrogram. Get time-wise maxes.
            mask_max_values = torch.max(spectrogram,dim=1)[0].unsqueeze(1)
            mask = spectrogram/mask_max_values
            mask = torch.where(mask > self.mask_ratio,1.0,0)

        elif target_type == 'relative_power':
            # Normalize the spectrogram. Get time-wise maxes.
            mask_max_values = torch.max(spectrogram,dim=1,keepdim=True)[0]
            mask = spectrogram/mask_max_values
            mask = torch.where((mask*mask) > self.mask_ratio,1.0,0)

        return mask


    def to_signal(self,mag,phase,window,hop_length,win_length,original_len = None):
        # Reconstruct the complex spectrogram from magnitude and phase
        reconstructed_stft = mag * torch.exp(1j * phase)
        
        # Perform ISTFT
        reconstructed_signal_batch = torch.istft(reconstructed_stft, n_fft=win_length, 
                                                 hop_length=hop_length, win_length=win_length, 
                                                 window=window,length=original_len)
        return reconstructed_signal_batch

