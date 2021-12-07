import glob
import librosa
import numpy as np

from tools.features_log_mel_bands import feature_extraction

from tools.file_io import load_yaml_file
from tools.argument_parsing import get_argument_parser
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from functools import partial

executor = ProcessPoolExecutor(max_workers=cpu_count())


def wav_to_mel(wav_file_path):
    
    args = get_argument_parser().parse_args(args=[])   
    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    settings_audio = settings['dataset_creation_settings']['audio']
    settings_features = settings['feature_extraction_settings']

    y = librosa.load(path=wav_file_path, sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])[0]
    print("wav")
    print(y)

    mel = feature_extraction(y, **settings_features['process'])
    print("feature")
    print(mel)

    return mel





