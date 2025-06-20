import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# --- New: Load Arousal and Valence Labels ---
def load_arousal_valence_labels(arousal_csv_path, valence_csv_path):
    """
    Loads arousal and valence data from CSVs and organizes them by song_id and segment.
    
    Returns:
        dict: A dictionary where keys are song_ids (str) and values are
              DataFrames with columns like 'timestamp_ms', 'arousal', 'valence'.
    """
    arousal_df = pd.read_csv(arousal_csv_path)
    valence_df = pd.read_csv(valence_csv_path)

    # Rename 'song_id' to 'file_id' for consistency with audio filenames
    arousal_df = arousal_df.rename(columns={'song_id': 'file_id'})
    valence_df = valence_df.rename(columns={'song_id': 'file_id'})

    # Melt the dataframes to have 'timestamp_ms' as a column
    arousal_melted = arousal_df.melt(id_vars=['file_id'], 
                                     var_name='sample_ms_str', 
                                     value_name='arousal')
    valence_melted = valence_df.melt(id_vars=['file_id'], 
                                     var_name='sample_ms_str', 
                                     value_name='valence')

    # Extract numeric timestamp from 'sample_ms_str' (e.g., 'sample_15000ms' -> 15000)
    arousal_melted['timestamp_ms'] = arousal_melted['sample_ms_str'].str.extract('(\d+)').astype(int)
    valence_melted['timestamp_ms'] = valence_melted['sample_ms_str'].str.extract('(\d+)').astype(int)

    # Merge arousal and valence DataFrames
    combined_df = pd.merge(arousal_melted, valence_melted, 
                           on=['file_id', 'timestamp_ms'], how='inner')
    
    # Sort by file_id and timestamp for consistency
    combined_df = combined_df.sort_values(by=['file_id', 'timestamp_ms'])

    # Group by file_id and store segments as DataFrames in a dictionary
    labels_by_song_segment = {
        str(file_id): group[['timestamp_ms', 'arousal', 'valence']].reset_index(drop=True)
        for file_id, group in combined_df.groupby('file_id')
    }
    
    print(f"Loaded labels for {len(labels_by_song_segment)} songs.")
    return labels_by_song_segment

# --- 2. Time-Aligned Feature Extraction ---
def extract_segmented_audio_features(audio_path, segment_length_ms=500, sr_target=22050):
    """
    Extracts audio features for fixed-length segments of an audio file.
    
    Args:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Length of each segment in milliseconds.
        sr_target (int): Target sampling rate for loading the audio.
        
    Returns:
        tuple: (list of feature arrays, list of start times in seconds)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr_target)
        
        # Convert segment length from ms to samples
        segment_length_samples = int(sr * (segment_length_ms / 1000.0))
        
        features_list = []
        segment_start_times_sec = []

        # Iterate through audio in segments
        for i in range(0, len(y) - segment_length_samples + 1, segment_length_samples):
            segment = y[i : i + segment_length_samples]
            
            # Ensure segment is not empty
            if len(segment) == 0:
                continue

            # Extract features for the current segment
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            
            # Tempo is usually for a whole track, or larger segments.
            # For 500ms, it might not be stable. Let's include it but be aware.
            # If a segment is too short for beat_track, it might fail.
            # Handle cases where tempo might not be computable for very short segments
            try:
                tempo, _ = librosa.beat.beat_track(y=segment, sr=sr, start_bpm=120, units='bpm')
            except:
                tempo = 0 # Default to 0 or np.nan if tempo cannot be reliably estimated

            rms = librosa.feature.rms(y=segment)
            zcr = librosa.feature.zero_crossing_rate(y=segment)

            # Take the mean of each feature over the segment's frames
            # Be careful with dimensions: mfcc, chroma, spectral_contrast, rms, zcr usually return (n_features, n_frames)
            # You want the mean across frames (axis=1).
            # If a feature calculation returns a single value (e.g., tempo), concatenate it directly.
            
            # Ensure all features have at least one frame to compute mean
            if mfcc.shape[1] > 0 and chroma.shape[1] > 0 and \
               spectral_contrast.shape[1] > 0 and rms.shape[1] > 0 and zcr.shape[1] > 0:
                
                segment_features = np.concatenate([
                    mfcc.mean(axis=1),
                    chroma.mean(axis=1),
                    spectral_contrast.mean(axis=1),
                    np.array([tempo]), # Tempo is a scalar, make it an array
                    rms.mean(axis=1),
                    zcr.mean(axis=1)
                ])
                features_list.append(segment_features)
                segment_start_times_sec.append(i / sr)
            else:
                print(f"Skipping segment starting at {i/sr:.2f}s due to insufficient frames for feature extraction.")

        return features_list, segment_start_times_sec
    except Exception as e:
        print(f"Error extracting segmented features from {audio_path}: {e}")
        return [], []

# --- Main Execution Loop ---
def main():
    arousal_csv_path = "C:/Users/yingx/OneDrive/Documents/Coding/Capstone/arousal.csv"
    valence_csv_path = "C:/Users/yingx/OneDrive/Documents/Coding/Capstone/valence.csv"
    audio_data_dir = "C:/Users/yingx/OneDrive/Documents/Coding/Capstone/data/MEMD_audio"

    # Load arousal and valence labels for all songs and segments
    labels_by_song_segment = load_arousal_valence_labels(arousal_csv_path, valence_csv_path)

    all_features = []
    all_arousal_labels = []
    all_valence_labels = []

    # Iterate through each labeled song and its segments
    for file_id, segment_labels_df in labels_by_song_segment.items():
        audio_filename = f"{file_id}.mp3" # Assuming filenames are '2.mp3', '3.mp3' etc.
        audio_path = os.path.join(audio_data_dir, audio_filename)

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}. Skipping.")
            continue

        print(f"\nProcessing {audio_filename} for segmented features...")
        features_per_segment, segment_start_times_sec = extract_segmented_audio_features(audio_path, segment_length_ms=500)

        # Match extracted features to labels by time
        for i, features_array in enumerate(features_per_segment):
            segment_start_sec = segment_start_times_sec[i]
            segment_start_ms = int(segment_start_sec * 1000)

            # Find the closest matching label in the DataFrame
            # Your CSV labels are at intervals like 15000ms, 15500ms etc.
            # Match the feature segment's start_time_ms to the timestamp_ms in the labels_df
            
            # Find the row in segment_labels_df that is closest to segment_start_ms
            # You might need to adjust this logic depending on exact timing alignment
            # A simple approach: find the label whose timestamp_ms matches segment_start_ms exactly
            # or is within a small tolerance.
            
            # For simplicity, let's assume your 'sample_Xms' perfectly align with 500ms segments
            # and that 'timestamp_ms' in the label CSV represents the *start* of the segment.
            
            # Find the label row for this specific segment timestamp
            matched_label_row = segment_labels_df[segment_labels_df['timestamp_ms'] == segment_start_ms]

            if not matched_label_row.empty:
                arousal_value = matched_label_row['arousal'].iloc[0]
                valence_value = matched_label_row['valence'].iloc[0]
                
                all_features.append(features_array)
                all_arousal_labels.append(arousal_value)
                all_valence_labels.append(valence_value)
            # else:
            #     print(f"No matching label found for {audio_filename} at {segment_start_ms}ms. Skipping segment.")


    # Convert lists to NumPy arrays
    X = np.array(all_features)
    y_arousal = np.array(all_arousal_labels)
    y_valence = np.array(all_valence_labels)

    if X.size == 0:
        print("No features extracted with matching labels. Please check audio paths, label CSVs, and timestamp alignment.")
        return

    print(f"\nTotal extracted feature segments: {X.shape[0]}")
    print(f"Features shape: {X.shape}")
    print(f"Arousal labels shape: {y_arousal.shape}")
    print(f"Valence labels shape: {y_valence.shape}")

    # --- Data Splitting and Scaling ---
    print("\nSplitting and scaling data...")
    # Split the dataset for both arousal and valence simultaneously
    X_train, X_test, y_arousal_train, y_arousal_test, y_valence_train, y_valence_test = train_test_split(
        X, y_arousal, y_valence, test_size=0.2, random_state=42
    )

    # Further split training data into training and validation
    X_train, X_val, y_arousal_train, y_arousal_val, y_valence_train, y_valence_val = train_test_split(
        X_train, y_arousal_train, y_valence_train, test_size=0.25, random_state=42 # 0.25 of 0.8 is 0.2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nData splitting and scaling complete.")
    print(f"Train features shape: {X_train_scaled.shape}")
    print(f"Train arousal labels shape: {y_arousal_train.shape}")
    print(f"Train valence labels shape: {y_valence_train.shape}")

    # --- Build the Multi-Output Regression Neural Network Model ---
    print("\nBuilding the Multi-Output Regression Neural Network model...")

    input_layer = Input(shape=(X_train_scaled.shape[1],))

    # Shared hidden layers for both outputs
    shared_hidden = Dense(256, activation='relu')(input_layer)
    shared_hidden = Dropout(0.3)(shared_hidden)
    shared_hidden = Dense(128, activation='relu')(shared_hidden)
    shared_hidden = Dropout(0.3)(shared_hidden)
    shared_hidden = Dense(64, activation='relu')(shared_hidden)
    shared_hidden = Dropout(0.3)(shared_hidden)

    # Separate output branches for Arousal and Valence
    arousal_output = Dense(1, activation='linear', name='arousal_output')(shared_hidden)
    valence_output = Dense(1, activation='linear', name='valence_output')(shared_hidden)

    # Create the model with multiple outputs
    model = Model(inputs=input_layer, outputs=[arousal_output, valence_output])

    # Compile the model - use 'mse' for regression tasks
    # You can also use separate loss weights if one output is more important
    model.compile(optimizer='adam',
                  loss={'arousal_output': 'mse', 'valence_output': 'mse'},
                  metrics={'arousal_output': 'mae', 'valence_output': 'mae'}) # MAE for easier interpretation

    model.summary()

    # --- Train the Model ---
    print("\nTraining the model...")
    history = model.fit(
        X_train_scaled,
        {'arousal_output': y_arousal_train, 'valence_output': y_valence_train},
        validation_data=(X_val_scaled, {'arousal_output': y_arousal_val, 'valence_output': y_valence_val}),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    print("Model training complete.")

    # --- Evaluate the Model ---
    print("\nEvaluating the model on the test set...")
    results = model.evaluate(
        X_test_scaled,
        {'arousal_output': y_arousal_test, 'valence_output': y_valence_test},
        verbose=0
    )

    print(f"Test Loss (Overall): {results[0]:.4f}")
    print(f"Test Arousal Loss (MSE): {results[1]:.4f}")
    print(f"Test Valence Loss (MSE): {results[2]:.4f}")
    print(f"Test Arousal MAE: {results[3]:.4f}")
    print(f"Test Valence MAE: {results[4]:.4f}")
    
    # Interpretation: A lower MAE means your model's predictions are closer to the true values.

    # --- Make Predictions (Example) ---
    print("\nMaking predictions on a sample from the test set...")
    sample_features = X_test_scaled[0:1] 
    predicted_arousal, predicted_valence = model.predict(sample_features)

    print(f"Sample's true arousal: {y_arousal_test[0]:.4f}, valence: {y_valence_test[0]:.4f}")
    print(f"Predicted arousal: {predicted_arousal[0][0]:.4f}, valence: {predicted_valence[0][0]:.4f}")


if __name__ == "__main__":
    main()