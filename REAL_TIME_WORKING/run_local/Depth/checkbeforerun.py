from missing import check_dataset, find_missing_files
import os
import glob
import json
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning DataSet Checking Script')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    args = parser.parse_args()
    find_missing_files(args.data_folder)
    check_dataset(args.data_folder)