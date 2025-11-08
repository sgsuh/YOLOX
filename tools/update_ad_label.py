"""
Create: 2022.04.18
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import json
import shutil
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root_fold",
                        default="")
    parser.add_argument("--dst_root_fold",
                        default="")
    parser.add_argument("--save_root_fold",
                        default="")
    parser.add_argument("--search_fold_list",
                        default=[""])

    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    src_root_fold = args.src_root_fold
    dst_root_fold = args.dst_root_fold
    save_root_fold = args.save_root_fold

    search_fold_list = args.search_fold_list

    src_fold_list = glob.glob(src_root_fold + '/*')
    src_fold_list.sort()

    for src_fold_path in src_fold_list:
        if not os.path.isdir(src_fold_path):
            continue

        src_file_list = glob.glob(src_fold_path + '/*.json')
        src_file_list.sort()

        for src_file_path in src_file_list:
            src_file_name = os.path.basename(src_file_path)

            is_find = False

            for search_fold_name in search_fold_list:
                search_fold_path = dst_root_fold + '/' + search_fold_name + '/images'
                search_file_path = search_fold_path + '/' + src_file_name

                if os.path.isfile(search_file_path):
                    print(search_file_path)
                    is_find = True
                    break

            if is_find:
                with open(search_file_path, 'r') as f_dst, open(src_file_path, 'r') as f_src:
                    dst_json_data = json.load(f_dst)
                    src_json_data = json.load(f_src)

                for idx, bbox_info in enumerate(src_json_data['shapes']):
                    if bbox_info['label'] == 'Pole2' or bbox_info['label'][:4] == 'Pole3':
                        src_json_data['shapes'].pop(idx)

                for bbox_info in dst_json_data['shapes']:
                    if bbox_info['label'] == 'Pole2' or bbox_info['label'][:4] == 'Pole3':
                        src_json_data['shapes'].append(bbox_info)

                with open(src_file_path, 'w') as f_out:
                    json.dump(src_json_data, f_out, indent = 2)
            else:
                fold_name = os.path.basename(src_fold_path)
                dst_fold_path = save_root_fold + '/' + fold_name

                if not os.path.isdir(dst_fold_path):
                    os.makedirs(dst_fold_path)

                file_name = os.path.splitext(os.path.basename(src_file_path))[0]
                dst_file_path = dst_fold_path + '/' + file_name + '.png'
                src_img_file_path = src_fold_path + '/' + file_name + '.png'

                shutil.copyfile(src_img_file_path, dst_file_path)
                            
