import glob
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
import torch
from PIL import Image

images = glob.glob('./../datasets/neon_data/neon_final_dataset_2/*.jpg')
scale = 0.70
cnt = 0

for fname in images:
    image = read_image(fname)
    result = torch.zeros(image.shape)
    H = image.shape[1]
    W = image.shape[2]
     
    # 타원 생성
    c_x = H/2
    c_y= W/2
    width = H/2 * scale
    height = W/2 * scale
    transform = transforms.Resize((int(2*height), int(2*width)))
    
    x = int(c_x - width)
    y = int(c_y - height)
    w = int(2 * width)
    h = int(2 * height)
    
    image = transform(image.type(torch.float32)/255.)
    # matrix = torch.tensor([[width ** 2, 0], [0, height ** 2]])
    # 타원 밖에 있는 텐서를 0으로 처리
    # image[0][torch.matmul(image[0] - center, matrix) > 0] = 0
    # image[1][torch.matmul(image[1] - center, matrix) > 0] = 0
    # image[2][torch.matmul(image - center, matrix) > 0] = 0
    # image = image[:, x:x+w, y:y+h]
    result[:, x:x+w, y:y+h] = image * 255
    
    image = result.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(f'/home/capston/instant-ngp/points_data/datasets/neon_data/neon_final_dataset_2_cropped/{cnt}.jpg')
    cnt += 1
    
   
