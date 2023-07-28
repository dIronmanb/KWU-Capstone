import time
import random
from tqdm import tqdm
import os
import torch
# from options.train_options import TrainOptions
# from data import create_dataset #! 광자데이터셋, 이미지데이터셋
# from models import create_model #! 제작하기

from options.train_options import parse_opt
from models.cycle_gan_model import CycleGANModel
from models.utils import get_filepaths
from models.volume_rendering_torch import volume_rendering, load_binary_file

#* Instruction #
#* python train.py 
#* --lambda_identity 0.0 
#* --pool_size 1 
#* --n_epochs 400 
#* --save_epoch_freq 10 
#* --print_freq 10 
#* --a_photon_dataset_path /home/capston/instant-ngp/points_data/lego 
#* --b_image_dataset_path /home/capston/instant-ngp/points_data/K_Army
#* --name lego_army (가중치를 저장할 디렉토리)

opt = parse_opt()   
A_points_paths = get_filepaths(opt.a_photon_dataset_path)
B_images_paths = get_filepaths(opt.b_image_dataset_path)
bigger = len(A_points_paths) if len(A_points_paths) < len(B_images_paths) else len(B_images_paths)
print(f'The number of training photons = {len(A_points_paths)}')
print(f'The number of training images = {len(B_images_paths)}')


model = CycleGANModel(opt)
model.load_networks(opt.epoch)
model.setup(opt) #! 여기는 나중에 다시 수정


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

total_iters = 0
#! error_epoch = 0
for epoch in range(int(opt.epoch), opt.n_epochs+1):
    error_epoch = epoch
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    model.update_learning_rate()
    
    for A_points_path, B_image_path in zip(A_points_paths, random.choices(B_images_paths, k=len(A_points_paths))):
        model.set_input(A_points_path, B_image_path)
        model.optimize_parameters()
        total_iters += 1
        epoch_iter += 1
        
        if total_iters % opt.print_freq == 0:
            losses = model.get_current_losses()
            result = f"(losses) G_A:{losses['G_A']:g} / D_A:{losses['D_A']:g} / cycle_A:{losses['cycle_A']:g} / "\
                            f"G_B:{losses['G_B']:g} / D_B:{losses['D_B']:g} / cycle_B:{losses['cycle_B']:g}\n"
            print("epoch : ",epoch,result)
            
            
            
            #TODO: 매 loss를 list 등에 담아서 임의의 디렉토리에 저장하기
            #TODO: 발표 시 loss 결과를 보여주기 위함
            #* TensorBoard에 띄우기 (나중에)
            #*
            #*
            #*
            #* #############################
            
        # if total_iters % opt.save_latest_freq == 0:
        #     print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #     model.save_networks(save_suffix) #! save networks
        # time.sleep(1)
    
    print(f"one epoch learning time:{time.time() - epoch_start_time:.3f}")        
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #model.save_networks('latest')
        model.save_networks(epoch)

# for epoch in range(int(opt.epoch), opt.n_epochs):
#     try:
#         error_epoch = epoch
#         epoch_start_time = time.time()
#         iter_data_time = time.time()
#         epoch_iter = 0
#         model.update_learning_rate()
        
#         for A_points_path, B_image_path in zip(A_points_paths, random.choices(B_images_paths, k=len(A_points_paths))):
#             model.set_input(A_points_path, B_image_path)
#             model.optimize_parameters()
#             total_iters += 1
#             epoch_iter += 1
            
#             if total_iters % opt.print_freq == 0:
#                 losses = model.get_current_losses()
#                 result = f"(losses) G_A:{losses['G_A']:g} / D_A:{losses['D_A']:g} / cycle_A:{losses['cycle_A']:g} / "\
#                                 f"G_B:{losses['G_B']:g} / D_B:{losses['D_B']:g} / cycle_B:{losses['cycle_B']:g}\n"
#                 print("epoch : ",epoch,result)
                
                
                
#                 #TODO: 매 loss를 list 등에 담아서 임의의 디렉토리에 저장하기
#                 #TODO: 발표 시 loss 결과를 보여주기 위함
#                 #* TensorBoard에 띄우기 (나중에)
#                 #*
#                 #*
#                 #*
#                 #* #############################
                
#             # if total_iters % opt.save_latest_freq == 0:
#             #     print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#             #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#             #     model.save_networks(save_suffix) #! save networks
#             time.sleep(1)
        
#         print(f"one epoch learning time:{time.time() - epoch_start_time:.3f}")        
#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             #model.save_networks('latest')
#         model.save_networks(epoch)
        
#     except Exception as e:
#         print(e)
#         print("Restarting the script")
        
#         torch.cuda.empty_cache()
#         model = CycleGANModel(opt)
#         error_epoch -= (error_epoch % opt.save_epoch_freq)
#         opt.epoch = error_epoch
#         model.load_networks(error_epoch-1)
#         model.setup(opt) #! 여기는 나중에 다시 수정



'''
import time
import torch

while True:
    try:
        # 실행할 코드
        # ...

        # 스크립트가 무한 루프에 머물지 않도록 일시 중지
        time.sleep(1)  # 1초

    except Exception as e:
        print("An error occurred:", str(e))
        print("Restarting the script...")

        # GPU 메모리 해제
        torch.cuda.empty_cache()

        continue  # 스크립트 다시 실행
'''
