import time
import random
from tqdm import tqdm
import os
import torch
import numpy as np

# from options.train_options import TrainOptions
# from data import create_dataset #! 광자데이터셋, 이미지데이터셋
# from models import create_model #! 제작하기

from options.train_options import parse_opt
from models.cycle_gan_model import CycleGANModel
from models.utils import get_filepaths

# import matplotlib.pyplot as plt
from PIL import Image

import re

'''
    python final_model/save_generated_images.py \
    --lambda_identity 0.0 \
    --pool_size 1 \
    --a_photon_dataset_path /home/capston/instant-ngp/points_data/3d_points/lego \
    --b_image_dataset_path /home/capston/instant-ngp/points_data/datasets/fire_data/fire_final_dataset_1 \
    --name mlp64_fire1 \
    --epoch 70
'''


def extract_numbers(string):
    numbers = re.findall(r'\d+', string)
    return [int(num) for num in numbers]

#* pool_size = 1

opt = parse_opt()

A_points_paths = get_filepaths(opt.a_photon_dataset_path)
B_images_paths = get_filepaths(opt.b_image_dataset_path)

A_points_paths.sort(key=lambda x:extract_numbers(x[-13:])) #! 이름들 정렬
print([name[-13:] for name in A_points_paths]) #! 이름을 정렬해도 뭔가 이상하다


model = CycleGANModel(opt)

model.setup(opt)
model.load_networks(opt.epoch)
model.netG_A.eval()



for view in range(len(A_points_paths)):
    model.set_input(A_points_paths[view], B_images_paths[0])
    model.forward()

    result = (model.fake_B_image * 255).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)
    
        
    #! pixel값이 10인 부분은 제거 (노이즈 제거)
    # image = Image.fromarray(result)
    result[np.where(result[...,:3] < 50)] = 0
    image = Image.fromarray(result.astype(np.uint8))

    # 이미지를 'rgba' 형식으로 변환
    image_rgba = image.convert('RGBA')

    # 0인 픽셀 값을 가진 픽셀의 알파 채널을 0으로 설정
    data = np.array(image_rgba)
    data[(data == [0, 0, 0, 255]).all(axis=2)] = [0, 0, 0, 0]
    image_rgba = Image.fromarray(data)

    # 이미지 저장
    image_rgba.save(f'/home/capston/instant-ngp/points_data/final_model/generated_images/lego_blue/r_{view}.png')
    
    
    # image = Image.fromarray(result)
    # image.save(f'/home/capston/instant-ngp/points_data/final_model/generated_images/chair_fire_01/r_{view}.png')
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB 단위로 변환
    print(f"현재 할당된 GPU 메모리: {allocated_memory} GB")
    
    # model 클리어
    model.clear()
    
