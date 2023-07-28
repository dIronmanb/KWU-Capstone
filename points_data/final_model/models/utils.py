import argparse
import torch
import numpy as np
import random

import os

#! 이 부분 알아봐야함
def fix_seed(random_seed):
    '''
    이 과정을 통해 학습을 진행할 때, random한 경우를 배제하여 동일한 결과가 나오게 한다.
    따라서 다른 변수를 바꿔보면서 실험을 진행할 수 있다.
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_filepaths(file_dir:str):
    return [os.path.join(file_dir, filename) for filename in os.listdir(file_dir)]



if __name__ == '__main__':
    result = get_filepaths('/home/capston/instant-ngp/points_data/lego')
    print(result)


