from flask import Flask, Response, jsonify, request, json, send_file, abort
from flask_cors import CORS
from swagger_ui import api_doc, flask_api_doc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import os
import asyncio
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import threading

# Create api
app = Flask(__name__)

# Create cors
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# S³ownik przechowuj¹cy blokady
locks = {}

# Asynchroniczna funkcja pomocnicza
async def async_generate_mel_spectrogram(file_path, png_file_path, lock_unique_name):
    # Sprawdzamy, czy blokada o unikalnej nazwie ju¿ istnieje
    if lock_unique_name not in locks:
        # Jeœli nie istnieje, tworzymy now¹ blokadê
        locks[lock_unique_name] = threading.Lock()

    # Pobieramy blokadê zwi¹zana z unikaln¹ nazw¹
    lock = locks[lock_unique_name]

    # Próba zablokowania blokady
    if lock.acquire(timeout=1):
        try:
            # Tutaj umieszczamy kod, który wymaga ochrony
            loop = asyncio.get_running_loop()
            # U¿ycie ThreadPoolExecutor do uruchomienia funkcji synchronicznej w asynchronicznym kontekœcie
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, generate_mel_spectrogram, file_path, png_file_path, lock_unique_name)
                return result
        finally:
            # Zawsze nale¿y zwolniæ blokadê, nawet jeœli coœ pójdzie nie tak
            lock.release()

# Synchroniczna funkcja do generowania spectrogramu
def generate_mel_spectrogram(file_path, png_file_path, lock_unique_name):
    try:
        # lock_manager.acquire(file_path)
        # with lock_manager:
        y, sr = librosa.load(file_path, sr=None) # Za³aduj plik audio
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        # plt.switch_backend(file_path)
        plt.figure(figsize=(25.6, 5.12))
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.title(Path(file_path).stem)
        plt.tight_layout()
        plt.savefig(png_file_path)
        plt.close()

        return f"Mel spectrogram saved to {png_file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

def process_boolean_param(value):
    """
    Converts a string parameter to a boolean.

    Args:
        value (str): A string representing a boolean, expected to be "true" or "false".
    
    Returns:
        bool: The boolean value of the string.
    """

    true_values = {True, "true", "1", "t", "yes"}
    false_values = {False, "false", "0", "f", "no"}

    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


@app.route('/api/get-mel-frequency-spectrogram', methods=['GET'])
async def get_mel_frequency_spectrogram():
    data = request.args

    try: 
        file_path = data['filePath']
    except KeyError: 
        file_path = None

    if file_path == None:
        try: 
            file_path = data.parms['FilePath']
        except KeyError: 
            file_path = None

    try: 
        remove_cache_entry = data['removeCacheEntry']
    except KeyError: 
        remove_cache_entry = False

    if remove_cache_entry == None:
        try: 
            remove_cache_entry = data.parms['RemoveCacheEntry']
        except KeyError: 
            remove_cache_entry = False

    path = Path(file_path)
    if not path.is_file():
        return Response('Resource not found', status=404)
    
    png_file_path = os.path.join(os.getenv("TEMP"), 'DjInTheHeaven.WebApi', 'get-mel-frequency-spectrogram', path.stem + '.png')
    path = Path(png_file_path)
    basename = os.path.basename(path)
    
    print(png_file_path)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    if process_boolean_param(remove_cache_entry) == True and os.path.exists(path):
        os.remove(path)
        path = Path(png_file_path)

    if path.is_file():
        return send_file(
            png_file_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=basename)

    await async_generate_mel_spectrogram(file_path, png_file_path, basename)

    path = Path(png_file_path)
    if not path.is_file():
        return Response('Resource not found', status=404)

    return send_file(
        png_file_path,
        mimetype="image/png",
        as_attachment=True,
        download_name=basename)

if __name__ == '__main__':
    # api_doc(app, config_path='./config/librosa-api.yaml', editor=True)
    # flask_api_doc(app, config_path='./conf/librosa-api.yaml', url_prefix='/api/doc', title='API doc')
    app.run()