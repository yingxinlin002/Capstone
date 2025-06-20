import librosa
import csv
import os
import soundfile as sf
import csv

def main():
    # Path to your audio file
    audio_path = "C:/Users/yingx/OneDrive/Documents/Coding/Capstone/data/MEMD_audio/2.mp3"

    # extract the data from first 5 audio files, where the mp3 files are named 1.mp3, 2.mp3, ..., 5.mp3
    for i in range(2, 6):   
        audio_path = f"C:/Users/yingx/OneDrive/Documents/Coding/Capstone/data/MEMD_audio/{i}.mp3"
        print(f"Processing audio file: {audio_path}")
        extract_features(audio_path)
        extract_waveform(audio_path)

def extract_features(audio_path):
    # Load audio file
    print("Loading audio file...")
    y, sr = librosa.load(audio_path)

    # Extract features
    print("Extracting features...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    # Prepare data for CSV (using mean values for each feature)
    data = {
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "chroma_mean": chroma.mean(axis=1).tolist(),
        "spectral_contrast_mean": spectral_contrast.mean(axis=1).tolist(),
        "tempo": [tempo],
        "rms_mean": rms.mean(axis=1).tolist(),
        "zcr_mean": zcr.mean(axis=1).tolist()
    }

    # Flatten the data for CSV
    csv_data = []
    header = []
    for key, values in data.items():
        if isinstance(values, list):
            for i, v in enumerate(values):
                header.append(f"{key}_{i+1}")
                csv_data.append(v)
        else:
            header.append(key)
            csv_data.append(values)

    # Write to CSV
    with open("extracted_features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(csv_data)

    print("Features written to extracted_features.csv")

def extract_waveform(input_audio_path):
    print(f"Loading audio file: {input_audio_path}...")
    y, sr = librosa.load(input_audio_path)
    output_wav = input_audio_path.replace('.mp3', '.wav')
    sf.write(output_wav, y, sr)
    print(f"Successfully extracted waveform and saved to {output_wav}")

    
if __name__ == "__main__":
    main()