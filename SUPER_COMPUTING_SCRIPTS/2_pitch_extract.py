import parselmouth
import numpy as np
import pickle

# Replace these paths with the paths to your input audio file and output pickle file
audio_path = "SOUND_FILES/5012_I.WAV"
pickle_path = "SOUND_FILES/5012_I_pitch.pkl"

# Function to extract and save pitch curve
def extract_and_save_pitch_curve(audio_path, pickle_path):
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch(time_step=0.1)  # Sample rate of 10 Hz
    pitch_values = pitch.selected_array['frequency']
    time_values = np.linspace(0, len(pitch_values) * 0.1, len(pitch_values))
    
    # Create a dictionary to store time and pitch values
    pitch_curve_data = {
        'time': time_values,
        'pitch': pitch_values
    }
    
    # Save the pitch curve data as a pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump(pitch_curve_data, f)

# Call the function to extract and save pitch curve
extract_and_save_pitch_curve(audio_path, pickle_path)
