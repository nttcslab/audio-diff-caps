import os
import argparse
import glob
import warnings
from tqdm import tqdm
import scaper


def main(dataset_dir):
    os.makedirs(dataset_dir+'/wav', exist_ok=True)
    jams_path_list = sorted(glob.glob(dataset_dir+'/jams/*.jams'))

    for jams_path in tqdm(jams_path_list):
        scaper.generate_from_jams(
            jams_path,
            audio_outfile=jams_path.replace('jams', 'wav')
            )


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str)
    args = parser.parse_args()
    main(dataset_dir = args.dataset_dir)