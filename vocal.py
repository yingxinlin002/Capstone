# Code source: Brian McFee
# License: ISC

##################
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import Audio
import os
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter, gaussian_filter1d  
def save_audio(audio_data, filename,sr):
        output_dir = "separated_tracks"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio_data,sr)
        print(f"Saved: {filepath}")
def spectral_masking_vocals(y, sr,
                            low_vocal_freq=70,
                            high_vocal_freq=5000,
                            vocal_attenuation_factor=0.2, # How much to attenuate non-vocal frequencies
                            mask_smoothing_sigma=1.5,     # Increased smoothing for softer transitions
                            rolloff_threshold_multiplier=0.01 # Factor for dynamic rolloff threshold
                        ):
    
        D = librosa.stft(y, hop_length=512)
        
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0]*2 - 2) # n_fft from D.shape[0]
        
        # 1. More flexible frequency range and softer masking edges
        # Create a sigmoid-like mask for smoother transitions
        center_freq = (low_vocal_freq + high_vocal_freq) / 2
        freq_range = high_vocal_freq - low_vocal_freq
        
        # A simple sigmoid-like curve (tanh) to create a soft frequency mask
        # This will create a mask that gradually goes from 0 to 1 within the specified range
        # You might need to adjust the 'slope' (e.g., 4 / freq_range) for desired steepness
        freq_mask_start = 1 / (1 + np.exp(-(freqs - low_vocal_freq) * (4 / (freq_range * 0.1))))
        freq_mask_end = 1 / (1 + np.exp((freqs - high_vocal_freq) * (4 / (freq_range * 0.1))))
        
        vocal_freq_mask_soft = freq_mask_start * freq_mask_end
        vocal_freq_mask_soft_2d = vocal_freq_mask_soft[:, np.newaxis]
        # Initial magnitude adjustment based on soft frequency mask
        vocal_magnitude = magnitude * vocal_freq_mask_soft_2d + \
                          magnitude * (1 - vocal_freq_mask_soft_2d) * vocal_attenuation_factor
        
        # 2. Refine the mask using spectral rolloff and dynamic threshold
        # Re-initialize vocal_mask based on the soft frequency mask to start with
        vocal_mask = vocal_freq_mask_soft_2d * np.ones_like(magnitude)

        # Calculate spectral rolloff
        # S=magnitude is correct here. You might want to experiment with different rolloff parameters.
        rolloff = librosa.feature.spectral_rolloff(sr=sr, S=magnitude, roll_percent=0.85) # Common roll_percent for vocal emphasis
        
        # Normalize rolloff dynamically based on its distribution in the current audio
        # Using mean and std for a more adaptive normalization
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        # Dynamic threshold based on rolloff characteristics
        # Adjust the multiplier to control sensitivity
        dynamic_rolloff_threshold = rolloff_mean + rolloff_std * rolloff_threshold_multiplier
        
        for i in range(magnitude.shape[1]):
            # If the rolloff is high (suggesting more high-frequency content, likely vocals)
            if rolloff[0, i] > dynamic_rolloff_threshold:
                vocal_mask[:, i] *= 1.2 # Boost vocal presence
            else:
                vocal_mask[:, i] *= 0.8 # Attenuate non-vocal areas
        
        # Ensure the mask values are within a reasonable range
        vocal_mask = np.clip(vocal_mask, 0.0, 1.5) # Allow slight boosts above 1 for emphasis

        # 3. Apply Gaussian smoothing for a less "choppy" sound
        vocal_mask = gaussian_filter(vocal_mask, sigma=mask_smoothing_sigma)
        vocal_mask = np.clip(vocal_mask, 0.1, 1.0) # Final clipping to prevent extreme values

        # Apply the final mask
        vocal_stft = magnitude * vocal_mask * np.exp(1j * phase)
        instrumental_stft = magnitude * (1 - vocal_mask * 0.8) * np.exp(1j * phase) # Adjust instrumental attenuation as needed
        
        vocals = librosa.istft(vocal_stft, hop_length=512)
        instrumental = librosa.istft(instrumental_stft, hop_length=512)
        
        return vocals, instrumental

# Load an example with vocals.
def main():
    y, sr = librosa.load('data/Pop1.wav')

    print("Load done")
    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))

   
    #######################################
    # Plot a 5-second slice of the spectrum
    idx = slice(*librosa.time_to_frames([10, 15], sr=sr))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                            
                            y_axis='log', x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()
    print("Plot done")
    ###########################################################
    # The wiggly lines above are due to the vocal component.
    # Our goal is to separate them from the accompanying
    # instrumentation.
    #

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.average,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimum
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)


    ##############################################
    # The raw filter output can be used as a mask,
    # but it sounds better if we use soft-masking.

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 5, 8
    power = 1

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    #S_foreground[np.abs(S_foreground)<3]=0
    print(librosa.amplitude_to_db(S_full, ref=np.max))
    S_foreground[np.log10(librosa.amplitude_to_db(S_full, ref=np.max))<512]=0
    ##########################################
    # Plot the same slice, but separated into its foreground and background

    # sphinx_gallery_thumbnail_number = 2
   
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                            y_axis='log', x_axis='time', sr=sr, ax=ax[0])
    ax[0].set(title='Full spectrum')
    ax[0].label_outer()

    librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                            y_axis='log', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Background')
    ax[1].label_outer()

    librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                            y_axis='log', x_axis='time', sr=sr, ax=ax[2])
    ax[2].set(title='Foreground')
    fig.colorbar(img, ax=ax)

    plt.show()
    ###########################################
    # Recover the foreground audio from the masked spectrogram.
    # To do this, we'll need to re-introduce the phase information
    # that we had previously set aside.

    y_foreground = librosa.istft(S_foreground * phase)
    #y_foreground, bg =spectral_masking_vocals(y_foreground,sr)
    y_foreground= save_audio(y_foreground,"vocal.wav",sr)
    print("Current working directory:", os.getcwd())

if __name__ == "__main__":
    main()