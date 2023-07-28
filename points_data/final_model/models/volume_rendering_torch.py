import cupy as cp
from time import time

try:
    flag = True
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    flag = False
    
from PIL import Image
import torch
import numpy as np
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
#! 함수 만들기


def load_binary_file(filename):

    desired_value_for_idx = 5
    desired_value = 20

    # start_time = time()

    with open(filename, "rb") as f:
        data = cp.fromfile(f, dtype=cp.float32)

    #! spp, resolution_x, resolution_y, size
    spp, resolution_x, resolution_y, cnt = data[0:4]
    spp, W, H, cnt = int(spp), int(resolution_x), int(resolution_y), int(cnt)
    data = data[4:] # renew data
    # print(f"spp:{spp}, resolution_y:{H}, resolution_x:{W}, size:{cnt}")

    #! n_alive_list, n_steps_list
    #* 맨 앞에 0추가
    n_alive_list = data[:cnt+1]
    n_steps_list = data[cnt+1:2*(cnt+1)]
    n_actual_steps_list = n_steps_list // n_alive_list
    n_actual_steps_list[0] = 0; # To solve DivisonZero

    n_alive_list = n_alive_list.astype(cp.uint32).tolist()
    n_steps_list = n_steps_list.astype(cp.uint32).tolist()
    n_actual_steps_list = n_actual_steps_list.astype(cp.uint32).tolist()

    data = data[2*(cnt+1):] # renew data

    #! indices
    indices = data[:H*W*desired_value_for_idx]
    data = data[H*W*desired_value_for_idx:]
    indices_list = [None] * cnt
    for idx in range(cnt):
        indices_list[idx] = indices[n_alive_list[idx]:n_alive_list[idx+1]].astype(cp.uint32)

    #! positions
    positions = data[:H*W*desired_value*3]
    data = data[H*W*desired_value*3:]
    positions_list = [None] * cnt
    for idx in range(cnt):
        positions_list[idx] = cp.reshape(positions[n_steps_list[idx]*3: n_steps_list[idx+1]*3],
                                        (n_actual_steps_list[idx+1], indices_list[idx].shape[0], 3))

    #! directions
    directions = data[:H*W*desired_value*3]
    data = data[H*W*desired_value*3:]
    directions_list = [None] * cnt
    for idx in range(cnt):
        directions_list[idx] = cp.reshape(directions[n_steps_list[idx]*3: n_steps_list[idx+1]*3],
                                        (n_actual_steps_list[idx+1], indices_list[idx].shape[0], 3))

    #! rgbs
    rgbs = data[:H*W*desired_value*3]
    data = data[H*W*desired_value*3:]
    rgbs_list = [None] * cnt
    for idx in range(cnt):
        rgbs_list[idx] = cp.reshape(rgbs[n_steps_list[idx]*3 : n_steps_list[idx+1]*3],
                                    (n_actual_steps_list[idx+1], indices_list[idx].shape[0], 3))

    #! densities
    densities = data[:H*W*desired_value]
    data = data[H*W*desired_value:]
    densities_list = [None] * cnt
    for idx in range(cnt):
        densities_list[idx] = cp.reshape(densities[n_steps_list[idx] : n_steps_list[idx+1]],
                                    (n_actual_steps_list[idx+1], indices_list[idx].shape[0], 1))
    #! distances
    dists = data[:H*W*desired_value]
    data = data[H*W*desired_value:]
    dists_list = [None] * cnt
    for idx in range(cnt):
        dists_list[idx] = cp.reshape(dists[n_steps_list[idx] : n_steps_list[idx+1]],
                                    (n_actual_steps_list[idx+1], indices_list[idx].shape[0], 1))

    # print(f"Loading Time:{time()-start_time:.2f}\n")

    return H, W, cnt, n_actual_steps_list, indices_list, positions_list, directions_list, rgbs_list, densities_list, dists_list

#! Coding Start
def volume_rendering(resolution_y,
                     resolution_x,
                     cnt,
                     n_actual_steps_list,
                     indices_list,
                     positions_list,
                     directions_list,
                     rgbs_list,
                     densities_list,
                     dists_list,
                     debugging_result = False,
                     debugging_T = False,
                     visualization = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #! Volume Rendering
    n_actual_steps_list = torch.tensor(n_actual_steps_list).to(device)
    #indices_list = torch.tensor(indices_list).to(device)
    #positions_list = torch.tensor(positions_list).to(device)
    #directions_list = torch.tensor(directions_list).to(device)
    #rgbs_list = torch.tensor(rgbs_list).to(device)
    #densities_list = torch.tensor(densities_list).to(device)
    #dists_list = torch.tensor(dists_list).to(device)

    #TODO: cp(=np) 변환 -> torch로
    # result = cp.zeros((resolution_y*resolution_x,3))
    result = torch.zeros((resolution_y*resolution_x,3)).to(device)

    #T = cp.ones((resolution_y*resolution_x, 1))
    T = torch.ones((resolution_y*resolution_x,1)).to(device)
    for i in range(cnt):

        indices = torch.tensor((cp.asnumpy(indices_list[i]).astype(np.int32))).to(device)
        positions = torch.tensor((cp.asnumpy(positions_list[i]).astype(np.float32))).to(device)
        directions = torch.tensor((cp.asnumpy(directions_list[i]).astype(np.float32))).to(device)#directions_list[i]
        rgbs = torch.tensor((cp.asnumpy(rgbs_list[i])).astype(np.float32)).to(device)#rgbs_list[i]
        densities = torch.tensor(cp.asnumpy(densities_list[i]).astype(np.float32)).to(device)#densities_list[i]
        dists = torch.tensor(cp.asnumpy(dists_list[i]).astype(np.float32)).to(device)#dists_list[i]



        #* 각 (n_steps_between_compaction, n_alives, tmp)에서의 n_steps에 대해 루프 적용
        for j in range(n_actual_steps_list[i+1]):
            # positions[j] # (n_alives, 3)
            #alpha = cp.ones((indices.shape[0],1)) - cp.exp(-densities[j] * dists[j]) # (n_alives, )
            alpha = torch.ones((indices.shape[0],1)).to(device) - torch.exp(-densities[j] * dists[j]) # (n_alives, )
            #print(T.shape,indices.long().shape,alpha.shape)
            weight = T[indices.long()] * alpha
            # tmp = cp.zeros((indices.shape[0], 3))
            result[indices.long()] += weight.repeat_interleave(3, axis=-1) * rgbs[j]
            # result[indices] += tmp
            T[indices.long()] *= (1 - alpha)

            #! Debugging T
            if flag and debugging_T:
                middle = torch.reshape(T, (resolution_y,resolution_x)) * 255
                plt.imshow(middle.astype(cp.uint8).get())
                plt.show()

            #! Debugging Result
            if flag and debugging_result:
                middle = torch.reshape(result, (resolution_y,resolution_x,3)) * 255
                plt.imshow(middle.astype(cp.uint8).get())
                plt.show()


    result = torch.reshape(result, (resolution_y,resolution_x,3))
    #TODO: cp(=np) 변환 -> torch로

    #! 시각화 하기
    # print(type(result))
    if flag and visualization:

        result_temp = result.to('cpu').detach().numpy()
        # torch type -> numpy()
        # result = result.detach().numpy() * 255
        result_temp *= 255
        plt.imshow(result_temp.astype(np.uint8))
        plt.show()

    return result



if __name__ == "__main__":


    filename = f"./points_0.bin"
    filename = f'/home/capston/instant-ngp/points_data/lego/points_0.bin'
    H, W, cnt, n_actual_steps_list, indices_list, positions_list, directions_list, rgbs_list, densities_list, dists_list = load_binary_file(filename)

    photon_data = [None] * cnt
    for i in range(cnt):
        indices = cp.repeat(indices_list[i][None, : , None], n_actual_steps_list[i+1], axis=0)
        photon_data[i] =  cp.concatenate([positions_list[i], directions_list[i], rgbs_list[i], densities_list[i], dists_list[i], indices], axis=-1)
        print(photon_data[i].shape)

        #! torch로 변환 (torch.from_numpy())
        #TODO1 photon_data = torch.from_numpy(photon_data[i].get())
        # tensor로 변환할 때, 원래 메모리를 상속받는다. (=as_tensor())
        #TODO2 photon_data = torch.from_numpy(cp.asnumpy(photon_data[i])
        # 데이터를 CPU로 복사하여 NumPy 배열로 변환
    informations = load_binary_file(filename)
    print("finish upload")
    #print(informations[4])
    result = volume_rendering(*informations,
                              debugging_T=False,
                              debugging_result=False,
                              visualization=True)
    #TODO3. pytorch용 volume rendering 진행해보기

