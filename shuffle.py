import os
import sys
import random
import json
import shutil
import subprocess
import argparse
import pandas as pd
import signal
import time
from datetime import datetime
import psutil

def send_signal_to_process_and_children(pid):
    # 获取父进程对象
    parent = psutil.Process(pid)
    # 向所有子进程发送 SIGINT
    for child in parent.children(recursive=True):  # 获取递归的所有子进程
        child.send_signal(signal.SIGINT)
    # 向父进程发送 SIGINT
    parent.send_signal(signal.SIGINT)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training and evaluation loop')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_loops', type=int, required=True, help='Number of loops to run')
    parser.add_argument('--sampling_ratio', type=int, required=True, help='Sampling ratio for the training set')
    parser.add_argument('--output_excel', type=str, required=True, help='Name of the output Excel file')
    return parser.parse_args()


def get_file_list(dataset_path):
    file_list_path = os.path.join(dataset_path, 'full_list.txt')
    with open(file_list_path, 'r') as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


def perform_sampling(dataset_path, num_loops, sampling_ratio, output_excel):
    file_list = get_file_list(dataset_path)
    previous_train_sets = set()  # 存储之前所有的训练集
    
    output_data = []

    for loop in range(num_loops):
        # Sample training set
        train_set_size = len(file_list) // sampling_ratio
        random.shuffle(file_list)
        train_set = set(file_list[:train_set_size])

        # Ensure the current train_set is not the same as any previous train set
        while any(train_set == prev_train_set for prev_train_set in previous_train_sets):
            random.shuffle(file_list)
            train_set = set(file_list[:train_set_size])

        previous_train_sets.add(frozenset(train_set))  # Add the current train set to the history

        test_set = set(file_list) - train_set
        
        # Save file names to respective text files
        with open(os.path.join(dataset_path, 'train_list.txt'), 'w') as f:
            for item in train_set:
                f.write(f"{item}\n")
        
        with open(os.path.join(dataset_path, 'test_list.txt'), 'w') as f:
            for item in test_set:
                f.write(f"{item}\n")
        
        with open(os.path.join(dataset_path, 'val_list.txt'), 'w') as f:
            for item in test_set:
                f.write(f"{item}\n")

        # Execute training command
        process = subprocess.Popen(f"ns-train tsplatter --data {dataset_path}", shell=True, env=os.environ)
        time.sleep(30)
        
        # Find the latest trained directory
        daily_stuff_dir = os.path.basename(dataset_path.rstrip('/'))
        tsplatter_dir = os.path.join(f"outputs/{daily_stuff_dir}/tsplatter/")
        latest_dir = max([d for d in os.listdir(tsplatter_dir) if os.path.isdir(os.path.join(tsplatter_dir, d))], key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H%M%S'))
        checkpoint_path = os.path.join(tsplatter_dir, latest_dir, 'nerfstudio_models', 'step-000000299.ckpt')

        while True:
            if os.path.exists(checkpoint_path):  # 如果文件存在
                time.sleep(3)
                send_signal_to_process_and_children(process.pid)
                process.wait()
                break  # 退出循环
            else:
                time.sleep(1)  # 每隔 1 秒钟检查一次

        # Execute evaluation command
        subprocess.run(
            f"ns-eval --load-config {os.path.join(tsplatter_dir, latest_dir, 'config.yml')} "
            f"--output-path {os.path.join(tsplatter_dir, latest_dir, 'eval.json')} "
            f"--render-output-path {os.path.join(tsplatter_dir, latest_dir, 'render')}",
            shell=True, env=os.environ)

        # Load eval results
        eval_json_path = os.path.join(tsplatter_dir, latest_dir, 'eval.json')
        with open(eval_json_path, 'r') as f:
            eval_results = json.load(f)

        psnr = eval_results['results']['psnr']
        ssim = eval_results['results']['ssim']
        lpips = eval_results['results']['lpips']

        # Move checkpoint to origin directory
        origin_dir = os.path.join(tsplatter_dir, latest_dir, 'origin')
        os.makedirs(origin_dir, exist_ok=True)
        shutil.move(checkpoint_path, origin_dir)

        # Execute smoothing command
        smoothing_command = f"python smoothing.py -s {dataset_path} --start_checkpoint {checkpoint_path} --feature_level 0 --topk 45 --encoder dino"
        subprocess.run(smoothing_command, shell=True, env=os.environ)

        # Execute evaluation command again for smoothed results
        subprocess.run(
            f"ns-eval --load-config {os.path.join(tsplatter_dir, latest_dir, 'config.yml')} "
            f"--output-path {os.path.join(tsplatter_dir, latest_dir, 'eval1.json')} "
            f"--render-output-path {os.path.join(tsplatter_dir, latest_dir, 'render1')}",
            shell=True, env=os.environ)

        # Load eval1 results
        eval1_json_path = os.path.join(tsplatter_dir, latest_dir, 'eval1.json')
        with open(eval1_json_path, 'r') as f:
            eval1_results = json.load(f)

        psnr1 = eval1_results['results']['psnr']
        ssim1 = eval1_results['results']['ssim']
        lpips1 = eval1_results['results']['lpips']

        # Calculate differences
        delta_psnr = psnr1 - psnr
        delta_ssim = ssim1 - ssim
        delta_lpips = lpips1 - lpips

        # Write results to Excel file
        output_data.append([
            psnr, ssim, lpips,
            psnr1, ssim1, lpips1,
            delta_psnr, delta_ssim, delta_lpips,
            ', '.join(train_set)
        ])

    # Save the data to Excel
    df = pd.DataFrame(output_data, columns=[
        'psnr', 'ssim', 'lpips', 'psnr1', 'ssim1', 'lpips1', 
        'delta_psnr', 'delta_ssim', 'delta_lpips', 'train_set_images'
    ])
    df.to_excel(output_excel, index=False)

    print(f'{output_excel} saved.')


if __name__ == '__main__':
    args = parse_arguments()
    perform_sampling(args.dataset_path, args.num_loops, args.sampling_ratio, args.output_excel)
