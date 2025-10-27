import sounddevice as sd
import soundfile as sf

def record_audio(filename="recorded_audio.wav", duration=5, fs=44100):
    """
    Records audio from the microphone and saves it as a WAV file.
    """
    print(f"🎤 Recording for {duration} seconds... Speak now!")

    # Record using sounddevice
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    print("✅ Recording complete! Saving file...")
    sf.write(filename, audio, fs)

    print(f"💾 Audio saved as '{filename}'")

if __name__ == "__main__":
    record_audio(duration=5)  # Record for 5 seconds
