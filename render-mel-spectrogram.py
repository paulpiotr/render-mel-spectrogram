from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Asynchroniczna funkcja pomocnicza
async def async_render_mel_spectrogram(mp3_file_path, png_file_path):
    # Tutaj umieszczamy kod, który wymaga ochrony
    loop = asyncio.get_running_loop()
    # U¿ycie ThreadPoolExecutor do uruchomienia funkcji synchronicznej w asynchronicznym kontekœcie
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, render_mel_spectrogram, mp3_file_path, png_file_path)
        return result

# Synchroniczna funkcja do generowania spectrogramu
def render_mel_spectrogram(mp3_file_path, png_file_path):
    try:
        # lock_manager.acquire(mp3_file_path)
        # with lock_manager:
        y, sr = librosa.load(mp3_file_path, sr=None) # Za³aduj plik audio
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        # plt.switch_backend(mp3_file_path)
        plt.figure(figsize=(25.6, 5.12))
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.title(Path(mp3_file_path).stem)
        plt.tight_layout()
        plt.savefig(png_file_path)
        plt.close()

        return f"Mel spectrogram saved to {png_file_path}"
    except Exception as e:
        return f"An error occurred: {e}"
