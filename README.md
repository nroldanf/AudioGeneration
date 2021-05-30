# AudioGeneration
Implementation of various conditioning and continuation music generation based on deep neural networks (Autoencoders, VAE, VQ-VAE, GAN, VAE-GAN, etc.)

- What kind of sounds are generated?
- Which features are used to train the model? (representations)
- Which neural networks architectures are used?


## Generation with raw audio (waveform)

__Challenges__:
- Long range dependencies are difficult to capture: different elements that are extracted from audio signals. e.g pitch, melody, timbre, harmony, rhythm.
- Computationally expensive. 
- Generation is low (takes a lot of time to generate)

Some examples of pretrained models are:

- WaveNet: Speech and Music
- Jukebox: 

## Generation with spectrograms

Preprocessing: Short-Time Fourier Transform applied to go to time-frequency domain.

Postprocessing: Generate spectrograms and then apply the Inverse transformation to synthetize a waveform.

Types of spectrogram:
- Vanilla spectrogram
- Log amplitude spectrogram
- Log-frequency-amplitude spectrogram
- Mel spectrograms (perceptually relevant for human beings)

Some examples of pretrained models are:

- MelNet
- DRUMGAN

Advantages:

- Temporal axis of spectrogram is more compact (less data). Interval between adjacent point is way larger.
- Capture longer time dependencies
- Computationally lighter than raw audio.

Challenges:

- More difficult to capture local patterns.
- Phase reconstruction can be problematic (we usually use magnitude of the transform for training and have to reconstruct the phase to get the raw audio waveform from the spectrogram)

You can't use MFCC to generate, because there is no inverse.

## Architectures

- GAN
- Autoencoders
- Variational Autoenconders (VAE)
- VQ-VAE (Vector Quantized Variational Autoencoder)
- LSTM (with MIDI files)

## Inputs for generation

### Conditioning

- Given an artist, genre, moods, etc.
- Open AI Jukebox, Text to Speech (text)

### Autonomous

- Generate sound without input 

### Continuation

- Use for instance, the start of a song or audio.
- LSTM


