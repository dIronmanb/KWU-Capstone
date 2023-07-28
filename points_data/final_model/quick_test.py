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

#* Instruction #
#* python quick_test.py 
#* --lambda_identity 0.0 
#* --pool_size 1 
#* --a_photon_dataset_path /home/capston/instant-ngp/points_data/lego 
#* --b_image_dataset_path /home/capston/instant-ngp/points_data/K_Army
#* --name lego_army (가중치를 저장할 디렉토리)
#* --epoch 80

'''
 잘되는 경우
 python final_model/train.py 
 --lambda_identity 0.0 
 --pool_size 1 
 --save_epoch_freq 5 
 --print_freq 10 
 --a_photon_dataset_path /home/capston/instant-ngp/points_data/3d_points/lego 
 --b_image_dataset_path /home/capston/instant-ngp/points_data/datasets/fire_data/fire_final_dataset_1 
 --name mlp64_fire1 
 --n_epochs 300 
 --epoch 105 
 --gpu_ids 1

'''

'''
    python final_model/quick_test.py \
    --lambda_identity 0.0 \
    --pool_size 1 \
    --a_photon_dataset_path /home/capston/instant-ngp/points_data/3d_points/lego 
    --b_image_dataset_path /home/capston/instant-ngp/points_data/datasets/fire_data/fire_final_dataset_1 \
    --name mlp64_fire1 \
    --epoch 70
'''

opt = parse_opt()
view = 7

A_points_paths = get_filepaths(opt.a_photon_dataset_path)
B_images_paths = get_filepaths(opt.b_image_dataset_path)

model = CycleGANModel(opt)

model.setup(opt)
model.load_networks(opt.epoch)

model.set_input(A_points_paths[view], B_images_paths[0])
model.netG_A.eval()

model.forward()

result = (model.fake_B_image * 255).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)

real_A_image = (model.real_A_image_buffer.query() * 255).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)


# #! result 변형 (H, W, 3) -> (H, W, 3)
# result = np.concatenate((result, np.ones((result.shape[0], result.shape[1], 1))*255), axis=-1)
#! pixel값이 10인 부분은 제거
result[np.where(result[...,:3] < 2)] = 0
image = Image.fromarray(result.astype(np.uint8))
# image.show()

# 이미지를 'rgba' 형식으로 변환
image_rgba = image.convert('RGBA')

# 0인 픽셀 값을 가진 픽셀의 알파 채널을 0으로 설정
data = np.array(image_rgba)
data[(data == [0, 0, 0, 255]).all(axis=2)] = [0, 0, 0, 0]
image_rgba = Image.fromarray(data)

image_rgba.show()

# 이미지 저장
# image_rgba.save('/home/capston/instant-ngp/points_data/final_model_copy/generated_images/lego_01/converted_image.png')


#! 일단 중지
# real_A_image = Image.fromarray(real_A_image)
# real_A_image.show()
