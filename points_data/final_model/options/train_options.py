import argparse


def parse_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hyper parameter
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    
    # gan_mode
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    # pool_size: 이미지 버퍼 (모델 안에 이미지를 계속 담음)
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    # lr
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    # beta
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    
    #* 있어도 되고, 없어도 그만 (지우진 X)
    # 224 x 224 x 3 x 64 // 10 * 2
    parser.add_argument('--photon_batch_size', type=int, default=1926758, help='how many photon data is calculated')
    # 224 x 224 x 3 x 64 // (800 x 800 x 3)
    parser.add_argument('--image_batch_size', type=int, default=5, help='how many image data is calculated')
    
    
    #TODO 나중에 데이터셋을 final_model/dataset에
    #TODO A_photon(O), A_image(있어도 되고, 없어도 그만), B_photon, B_image(O)
    parser.add_argument('--a_photon_dataset_path', type=str, default='/home/capston/instant-ngp/points_data/lego')
    parser.add_argument('--b_photon_dataset_path', type=str, default='')
    parser.add_argument('--a_image_dataset_path', type=str, default='')
    parser.add_argument('--b_image_dataset_path', type=str, default='/home/capston/instant-ngp/points_data/lego_B_style')

    # epoch 관련   
    # n_epoch: 총 학습할 에폭 횟수 
    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs with the initial learning rate')
    #????
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    #????
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    
    # n_images x n_epochs => about 10,000
    #* loss log 출력 주기, 모델 저장 주기
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_latest_freq', type=int, default=100)
    parser.add_argument('--save_epoch_freq', type=int, default=1)
    
    # isTrain: 학습 or 테스트
    parser.add_argument('--isTrain', type=int, default=1, help='When you want to train model, input the number <1>.')
    #TODO: gpu_ids (0) -> GPU 2개 달면 알아서 할당이 되는가? (나중에 찍어보는 걸로)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--data_parallel', type=int, default=0, help='turn on to use multi gpus.')
    # 가중치 저장할 디렉토리
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    #??? name (help 읽어보기)
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    #??? preprocess는 쓰는 건지는 미정 (없어도 됨)
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    #! lr_policy (스케쥴러: learning rate 변경시키는 친구)
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    
    #* 일단 공식 코드에서 긁어온거라 따로 알아보진 X
    #??? continue_train()
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    
    #* 일단 공식 코드에서 긁어온거라 따로 알아보진 X  
    parser.add_argument("--gpu_setting", type=int, default=0) # 현재 사용 가능한 cuda number는 '0'
    parser.add_argument("--num_thread", type=int, default=24) #! 사용하고자하는 cuda thread의 개수를 설정(후에 코드 작성 후 확인)
    #! random seed (실험을 매번 동일하게 만들어줄때)
    parser.add_argument("--random_seed", type=int, default=0) # 일관된 결과를 확인해보면서 결과를 확인해보기 위해서 사용함.

    #* 일단 공식 코드에서 긁어온거라 따로 알아보진 X  
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--total_iters", type=int, default=100)
    parser.add_argument("--idt_val", type=float, default=0.5)

    #! parser에 있는 내용들을 멤버변수로 변환
    opt = parser.parse_args()
    
    # python3 train.py --lab 0 ... 0.
    
    return opt

'''
def parse_opt():

    # parser객체 생성
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train", type=bool, default=True) # train을 진행할 것인지를 위한 parser
    parser.add_argument("--gpu_setting", type=int, default=0) # 현재 사용 가능한 cuda number는 '0'
    parser.add_argument("--num_thread", type=int, default=24) #! 사용하고자하는 cuda thread의 개수를 설정(후에 코드 작성 후 확인)
    parser.add_argument("--random_seed", type=int, default=0) # 일관된 결과를 확인해보면서 결과를 확인해보기 위해서 사용함.

    # parser.add_argument("--lr", type=float, default=0.0002)
    # parser.add_argument("--beta1", type=float, default=0.5) #! 0.5는 일단 cyclegan 코드에서 사용해서 사용

    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--total_iters", type=int, default=100)

    parser.add_argument("--idt_val", type=float, default=0.5)
    opt = parser.parse_args()

    return opt
'''