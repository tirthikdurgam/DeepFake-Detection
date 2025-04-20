#!/usr/bin/env python
""" Downloads FaceForensics++ and Deep Fake Detection public data release
Example usage:
    see -h or https://github.com/ondyari/FaceForensics
"""
import argparse
import os
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join

# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

# Parameters
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

ALL_DATASETS = list(DATASETS.keys())
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all', choices=ALL_DATASETS + ['all'],
                        help='Dataset to download.')
    parser.add_argument('-c', '--compression', type=str, default='raw', choices=COMPRESSION,
                        help='Compression type.')
    parser.add_argument('-t', '--type', type=str, default='videos', choices=TYPE,
                        help='File type: videos, masks, or models.')
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Number of videos to download.')
    parser.add_argument('--server', type=str, default='EU', choices=SERVERS,
                        help='Download server.')
    
    args = parser.parse_args()
    server_url = {'EU': 'http://canis.vc.in.tum.de:8100/',
                  'EU2': 'http://kaldir.vc.in.tum.de/faceforensics/',
                  'CA': 'http://falas.cmpt.sfu.ca:8100/'}[args.server]
    
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/Deepfakes/models/'
    
    return args

def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(out_file):
        temp_file, temp_path = tempfile.mkstemp(dir=out_dir)
        os.close(temp_file)
        if report_progress:
            urllib.request.urlretrieve(url, temp_path, reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, temp_path)
        os.rename(temp_path, out_file)
    else:
        tqdm.write(f'WARNING: Skipping existing file {out_file}')

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rProgress: {percent}%, {progress_size // (1024 * 1024)} MB, {speed} KB/s, {int(duration)} seconds elapsed")
    sys.stdout.flush()

def main(args):
    print('By continuing, you confirm agreement to the FaceForensics terms of use:')
    print(args.tos_url)
    input('Press Enter to continue, or CTRL+C to exit.')

    datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    
    for dataset in datasets:
        dataset_path = DATASETS[dataset]
        output_path = join(args.output_path, dataset_path, args.compression, args.type)
        os.makedirs(output_path, exist_ok=True)
        
        if dataset in ['original_youtube_videos', 'original_youtube_videos_info']:
            print(f'Downloading {dataset}. This may take a while.')
            download_file(args.base_url + dataset_path, join(output_path, 'downloaded_videos.zip'), report_progress=True)
            continue
        
        print(f'Downloading {args.type} for dataset {dataset_path}')
        filelist_url = args.base_url + FILELIST_URL
        filelist = json.loads(urllib.request.urlopen(filelist_url).read().decode("utf-8"))
        
        if args.num_videos is not None and args.num_videos > 0:
            print(f'Downloading first {args.num_videos} videos.')
            filelist = filelist[:args.num_videos]
        
        dataset_videos_url = args.base_url + f'{dataset_path}/{args.compression}/{args.type}/'
        filelist = [filename + '.mp4' for filename in filelist]
        
        for filename in tqdm(filelist):
            download_file(dataset_videos_url + filename, join(output_path, filename))

if __name__ == "__main__":
    args = parse_args()
    main(args)
