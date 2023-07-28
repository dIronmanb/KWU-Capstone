import torch
import numpy as cp

from torchvision.io import read_image
from torchvision import transforms

import itertools
from .base_model import BaseModel

from .model import ResidualBlock, GeneratorMLP, Generator, Discriminator
from .image_pool import ImagePool, ImageReplayBuffer
from .criterion import GANLoss
from .utils import fix_seed
from .volume_rendering_torch import volume_rendering, load_binary_file
from .data_parallel import DataParallelModel, DataParallelCriterion
# from utils import *


class CycleGANModel(BaseModel):


    # ! 내가 정해주는 것들
    # -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    # -- self.model_names (str list):         define networks used in our training.
    # -- self.visual_names (str list):        specify the images that you want to display and save.
    # -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        fix_seed(opt.random_seed)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = GeneratorMLP(in_features=9, out_features=3)
        self.netG_B = GeneratorMLP(in_features=9, out_features=3)
        self.netD_A = Discriminator(n_layers=4, input_c=3, n_filter=32, kernel_size=3)
        self.netD_B = Discriminator(n_layers=4, input_c=3, n_filter=32, kernel_size=3)
        
        # define loss functions
        #! LSGAN, Vanilla, W-GAN, ...
        self.criterionGAN = GANLoss(opt.gan_mode)# define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        
        if self.opt.data_parallel:  # define discriminators
            # torch.nn.DataParallel
            self.netG_A = DataParallelModel(self.netG_A).cuda()
            self.netG_B = DataParallelModel(self.netG_B).cuda()
            self.netD_A = DataParallelModel(self.netD_A).cuda()
            self.netD_B = DataParallelModel(self.netD_B).cuda()
            self.criterionGAN = DataParallelModel(self.criterionGAN).cuda()
            self.criterionCycle = DataParallelModel(self.criterionCycle).cuda()
            self.criterionIdt = DataParallelModel(self.criterionIdt).cuda()

            # self.criterionGAN = DataParallelModel(self.criterionGAN).cuda()
            # self.criterionCycle = DataParallelCriterion(self.criterionCycle).cuda()
            # self.criterionIdt = DataParallelCriterion(self.criterionIdt).cuda()
        else:
            self.netG_A = self.netG_A.to(self.device)
            self.netG_B = self.netG_B.to(self.device)
            self.netD_A = self.netD_A.to(self.device)
            self.netD_B = self.netD_B.to(self.device)
            self.criterionGAN = self.criterionGAN.to(self.device)
            self.criterionCycle = self.criterionCycle.to(self.device)
            self.criterionIdt = self.criterionIdt.to(self.device)
            


        if self.isTrain:
            #* pool size는 알아서 결정하기
            #! pool_size는 (min)10 ~ (max)30까지
            self.real_A_image_buffer = ImageReplayBuffer(opt.pool_size)
            self.real_B_image_buffer = ImageReplayBuffer(opt.pool_size)
            self.fake_A_image_buffer = ImageReplayBuffer(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_image_buffer = ImageReplayBuffer(opt.pool_size)  # create image buffer to store previously generated images

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.photon_batch_size = opt.photon_batch_size #! 미정
        self.image_batch_size  = opt.image_batch_size #! 10


    #TODO: 이거 어디서 사용하는지 알아보기
    #! 여기도 수정 바람
    def set_input(self, A_phton_file_path, B_image_file_path):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #! 매 binary file마다 읽어주기 (데이터를 loop하면서)
        self.H, self.W, self.real_A_photon, real_A_image = self.load_photon_file(A_phton_file_path)
        real_B_image = self.load_image_file(B_image_file_path) #TODO: B style image 가져오기 (buffer로 가져오기)
        

        real_A_image = real_A_image.to(self.device) if not self.opt.data_parallel else real_A_image.cuda()
        real_B_image = real_B_image.to(self.device) if not self.opt.data_parallel else real_B_image.cuda()
        
        # print(f'\n\n\n\n{self.device}\n\n\n\n')
        self.real_A_image_buffer.append(real_A_image)
        self.real_B_image_buffer.append(real_B_image)

        #TODO: 나중에 LAM에 올리는 방법 참고 -> DataLoader에 올리는 방법 및 ...
        #TODO: GPU 단에 올리는 건 매우 부담이 큼


    def forward(self):
        #! real_A_photon, real_A_image, real_B_image
        #**********************************************************************#
        
        #* G_A(real_A)
        self.fake_B_rgbs, self.fake_B_image = self.forward_G(self.netG_A, self.real_A_photon)
        self.fake_B_photon = self.make_new_photon_data(self.real_A_photon, self.fake_B_rgbs)

        #* G_B(fake_B)
        self.rec_A_rgbs,  self.rec_A_image  = self.forward_G(self.netG_B, self.fake_B_photon)

        #* G_B(real_B) but G_B(fake_B)
        self.fake_A_rgbs, self.fake_A_image = self.forward_G(self.netG_B, self.fake_B_photon)
        fake_A_photon = self.make_new_photon_data(self.fake_B_photon, self.fake_A_rgbs)

        #* G_A(fake_A)
        self.rec_B_rgbs,  self.rec_B_image  = self.forward_G(self.netG_A, fake_A_photon)        
        
        # #########################################################################
        # allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB 단위로 변환
        # print(f"현재 할당된 GPU 메모리: {allocated_memory} GB")
        # #########################################################################

    def backward_D_basic(self, netD, real_image, fake_image):
        #* 묶음 단위로 학습하는 건 backward_D_A 및 backward_D_B에서 이미 만듦
        netD = netD.to(self.device)
        if self.opt.data_parallel:
            # Real
            pred_real = netD(real_image[None,...]) #! 여기서는 2D image 사용
            real_label = torch.ones(pred_real[0].shape).cuda()
            loss_D_real = self.criterionGAN(pred_real[0], real_label)
            loss_D_real = torch.mean(torch.tensor([loss_D_real_i_gpu.cuda(device=0) for loss_D_real_i_gpu in loss_D_real], requires_grad=True).cuda(device=0))
            # Fake
            pred_fake = netD(fake_image.detach()[None,...]) #* Generator backward() 차단
            fake_label = torch.zeros(pred_fake[0].shape).cuda()
            loss_D_fake = self.criterionGAN(pred_fake[0], fake_label)
            loss_D_fake = torch.mean(torch.tensor([loss_D_fake_i_gpu.cuda(device=0) for loss_D_fake_i_gpu in loss_D_fake], requires_grad=True).cuda(device=0))
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
        else:
            # Real
            pred_real = netD(real_image) #! 여기서는 2D image 사용
            real_label = torch.ones(pred_real.shape).to(self.device)
            loss_D_real = self.criterionGAN(pred_real, real_label)
            # Fake
            pred_fake = netD(fake_image.detach()) #* Generator backward() 차단
            fake_label = torch.zeros(pred_fake.shape).to(self.device)
            loss_D_fake = self.criterionGAN(pred_fake, fake_label)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.fake_B_image_buffer.append(self.fake_B_image)
        real_B = self.real_B_image_buffer.query(size=self.image_batch_size)
        fake_B = self.fake_B_image_buffer.query(size=self.image_batch_size)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.fake_A_image_buffer.append(self.fake_A_image)
        real_A = self.real_A_image_buffer.query(size=self.image_batch_size)
        fake_A = self.fake_A_image_buffer.query(size=self.image_batch_size)
        self.loss_D_B = self.backward_D_basic(self.netD_B, real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if self.opt.data_parallel:
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A_rgbs, _ = self.forward_G(self.netG_A, self.fake_B_photon)
                self.loss_idt_A = self.criterionIdt(self.flatten_rgbs(self.idt_A_rgbs), self.flatten_rgbs(self.fake_B_rgbs))
                self.loss_idt_A = torch.mean(torch.tensor(self.loss_idt_A, requires_grad=True).cuda(device=1)) * lambda_B * lambda_idt           
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B_rgbs, _  = self.forward_G(self.netG_B, self.real_A_photon)
                self.loss_idt_B = self.criterionIdt(self.flatten_rgbs(self.idt_B_rgbs), self.flatten_rgbs(self.extract_rgbs(self.real_A_photon)))
                self.loss_idt_B = torch.mean(torch.tensor(self.loss_idt_B, requires_grad=True).cuda(device=1)) * lambda_A * lambda_idt           
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0
        else:
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A_rgbs, _ = self.forward_G(self.netG_A, self.fake_B_photon)
                self.loss_idt_A = self.criterionIdt(self.flatten_rgbs(self.idt_A_rgbs), self.flatten_rgbs(self.fake_B_rgbs)) * lambda_B * lambda_idt            
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B_rgbs, _  = self.forward_G(self.netG_B, self.real_A_photon)
                self.loss_idt_B = self.criterionIdt(self.flatten_rgbs(self.idt_B_rgbs), self.flatten_rgbs(self.extract_rgbs(self.real_A_photon))) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0
        
        # tmp = self.netD_A(self.fake_B_image[None,...])
        # print("Tmp shape:", tmp[0].shape, tmp[0].device)
        # # print(tmp[0].device, tmp[0].shape, tmp[1].device, tmp[1].shape)
        # print(self.fake_B_image[None,...].shape)
        
        #! 가짜를 진짜 같게
        
        
        if self.opt.data_parallel:
            # GAN loss D_A(G_A(A)))
            #TODO 나중에 3개 이상 사용할 때에는 또 달라질 수도 있다. 
            #! GPU가 3개 이상일 때는 또 다르게 해주어야 함
            A_result = self.netD_A(self.fake_B_image[None,...])
            A_real_label = torch.ones(A_result[0].shape)
            A_real_label = A_real_label.to(self.device) if not self.opt.data_parallel else A_real_label.cuda()
            self.loss_G_A = self.criterionGAN(A_result[0], A_real_label)
            self.loss_G_A = torch.mean(torch.tensor([loss_G_A_i_gpu.cuda(device=0) for loss_G_A_i_gpu in self.loss_G_A], requires_grad=True).cuda(device=0))
            # GAN loss D_B(G_B(B))
            B_result = self.netD_B(self.fake_A_image[None,...])
            B_real_label = torch.ones(B_result[0].shape)
            B_real_label = B_real_label.to(self.device) if not self.opt.data_parallel else B_real_label.cuda()
            self.loss_G_B = self.criterionGAN(B_result[0], B_real_label)
            self.loss_G_B = torch.mean(torch.tensor([loss_G_B_i_gpu.cuda(device=0) for loss_G_B_i_gpu in self.loss_G_B], requires_grad=True).cuda(device=0))
        else:
            # GAN loss D_A(G_A(A))
            self.netD_A = self.netD_A.to(self.device)
            self.netD_B = self.netD_B.to(self.device)
            A_result = self.netD_A(self.fake_B_image)
            A_real_label = torch.ones(A_result.shape)
            A_real_label = A_real_label.to(self.device) if not self.opt.data_parallel else A_real_label.cuda()
            self.loss_G_A = self.criterionGAN(A_result, A_real_label)
            # GAN loss D_B(G_B(B))
            B_result = self.netD_B(self.fake_A_image)
            B_real_label = torch.ones(B_result.shape)
            B_real_label = B_real_label.to(self.device) if not self.opt.data_parallel else B_real_label.cuda()
            self.loss_G_B = self.criterionGAN(B_result, B_real_label)

        
        #! CycleLoss
        if self.opt.data_parallel:
            #! GPU가 3개 이상일 때는 또 다르게 해주어야 함
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.flatten_rgbs(self.rec_A_rgbs), self.flatten_rgbs(self.extract_rgbs(self.real_A_photon)))
            self.loss_cycle_A = torch.mean(torch.tensor([loss_cycle_A_i_gpu.cuda(device=0) for loss_cycle_A_i_gpu in self.loss_cycle_A], requires_grad=True).cuda(device=0)) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.flatten_rgbs(self.rec_B_rgbs), self.flatten_rgbs(self.fake_B_rgbs))
            self.loss_cycle_B = torch.mean(torch.tensor([loss_cycle_B_i_gpu.cuda(device=0) for loss_cycle_B_i_gpu in self.loss_cycle_B], requires_grad=True).cuda(device=0)) * lambda_B
        else:
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.flatten_rgbs(self.rec_A_rgbs), self.flatten_rgbs(self.extract_rgbs(self.real_A_photon))) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.flatten_rgbs(self.rec_B_rgbs), self.flatten_rgbs(self.fake_B_rgbs)) * lambda_B

        #! 모든 loss들을 더함
        # combined loss and calculate gradients
        if self.opt.data_parallel:
            self.loss_G = self.loss_G_A + \
                          self.loss_G_B + \
                          self.loss_cycle_A + \
                          self.loss_cycle_B + \
                          self.loss_idt_A.cuda(device=0) + \
                          self.loss_idt_B.cuda(device=0)
        else:
            self.loss_G = self.loss_G_A + \
                self.loss_G_B + \
                self.loss_cycle_A + \
                self.loss_cycle_B + \
                self.loss_idt_A + \
                self.loss_idt_B
                
        self.loss_G.backward()
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        #* self.fake_A, self.fake_B, self.rec_A, self.rec_B
        self.forward()      # compute fake images and reconstruction images.
    
        #* G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        #* D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


    def forward_G(self, netG, photon_data):

        netG = netG.to(self.device)

        #! photon_data -> photon_data.to(self.device)
        #! Multi GPU를 사용하다 보니 구현에 애를 먹는 듯. 차츰차츰 진행하자.
        T = torch.ones((self.H * self.W,1))
        weight = torch.ones((self.H * self.W,1))
        rendered_image = torch.zeros((self.H * self.W,3))

        
        T = T.to(self.device) if not self.opt.data_parallel else T.cuda()
        weight = weight.to(self.device) if not self.opt.data_parallel else weight.cuda()
        rendered_image = rendered_image.to(self.device) if not self.opt.data_parallel else rendered_image.cuda()
        
        new_rgbs = [None] * len(photon_data); cnt = 0
        
        for i_th_photon_data in photon_data:
            # print(f'\n{i_th_photon_data.shape}')
            i_th_photon_data = i_th_photon_data.to(self.device) if not self.opt.data_parallel else i_th_photon_data.cuda()
            n_steps, n_alives, n_features = i_th_photon_data.shape
            i_th_photon_data = torch.reshape(i_th_photon_data, (n_steps * n_alives, 12)) 

            # print(f'\n{i_th_photon_data.shape}')
            #! 일단 MLP에서 hidden layer의 node에 따라서 변경
            if self.photon_batch_size < n_steps * n_alives:
                tmp_rgbs = []
                for i in range(0, (n_steps * n_alives) // self.photon_batch_size, 1):
                    tmp_rgbs.append(netG(i_th_photon_data[i*self.photon_batch_size : (i+1)*self.photon_batch_size, 0:9]))
                else:
                    if (n_steps * n_alives) % self.photon_batch_size != 0:
                        tmp_rgbs.append(netG(i_th_photon_data[(i+1)*self.photon_batch_size:,0:9])) 

                #! GPU 2개에서 모든 데이터가 어떻게 들어가는지 알아봐야 한다.
                #! (ex1) cuda:0, cuda:1, cuda:0, cuda:1 
                #! (ex2) cuda:0, cuda:0, cuda:1, cuda:1 (일단 나는 이거 책정)
                if self.opt.data_parallel:
                    gpu_0_rgbs = torch.cat([rgbs[0] for rgbs in tmp_rgbs], dim = 0)
                    gpu_1_rgbs = torch.cat([rgbs[1] for rgbs in tmp_rgbs], dim = 0).cuda(device=0)
                    output_rgbs = torch.cat([gpu_0_rgbs, gpu_1_rgbs], dim = 0)
                else:
                    output_rgbs = torch.cat(tmp_rgbs, dim=0)
                    
            else:
                tmp_rgbs = netG(i_th_photon_data[:, 0:9]) # (n_steps * n_alive, 3)
                if self.opt.data_parallel:
                    gpu_0_rgbs = tmp_rgbs[0]
                    gpu_1_rgbs = tmp_rgbs[1].cuda(device=0)
                    output_rgbs = torch.cat([gpu_0_rgbs, gpu_1_rgbs], dim = 0)
                else:
                    output_rgbs = tmp_rgbs                
            
            #! inplace 문제 발생 수정           
            #* photon data -> new photon data (rgb만 바뀐)
            # i_th_photon_data_copy = i_th_photon_data.clone()
                        
            # i_th_photon_data_copy[..., 6:9] = output_rgbs
            
            # output_rgbs = torch.reshape(output_rgbs, (n_steps, n_alives, 3))
            # i_th_photon_data_copy = torch.reshape(i_th_photon_data_copy, (n_steps, n_alives, -1))

            # i_th_photon_data = i_th_photon_data_copy.clone()

            # new_photon_data[cnt] = i_th_photon_data
            # cnt += 1
            # * # * # * # * # * # * # * # * # * # * # * # * # 
            
            output_rgbs = torch.reshape(output_rgbs, (n_steps, n_alives, 3))
            new_rgbs[cnt] = output_rgbs
            cnt += 1
            i_th_photon_data = torch.reshape(i_th_photon_data, (n_steps, n_alives, -1))
            
            # volume rendering
            for j in range(n_steps):
                # alpha = torch.ones((i_th_photon_data.shape[1], 1)).to(self.device) - torch.exp(-i_th_photon_data[j,:,9] * i_th_photon_data[j,:,10])
                alpha = 1.0 - torch.exp(-i_th_photon_data[j,:,9] * i_th_photon_data[j,:,10])                               
                weight = T[i_th_photon_data[0,:,-1].long()] * alpha[...,None]
                rendered_image[i_th_photon_data[0,:,-1].long()] = rendered_image[i_th_photon_data[0,:,-1].long()] + weight.repeat_interleave(3, dim=-1) * output_rgbs[j]
                T[i_th_photon_data[0,:,-1].long()] = T[i_th_photon_data[0,:,-1].long()] * (1. - alpha)[..., None]
        
        rendered_image = rendered_image.reshape(self.H, self.W, 3).permute(2,1,0)
        return new_rgbs, rendered_image
    
    def make_new_photon_data(self, origin_photon_data, new_rgbs):
        new_photon_data = [None] * len(origin_photon_data)
        
        for i in range(len(new_photon_data)):
            new_photon_data[i] = origin_photon_data[i].clone()
            new_photon_data[i][...,6:9] = new_rgbs[i].clone()
            new_photon_data[i] = new_photon_data[i].to(self.device) if not self.opt.data_parallel else new_photon_data[i].cuda()

        return new_photon_data
    

    def get_rgbs(self, photon):
        result = torch.cat([torch.reshape(i[...,6:9], (-1,3)) for i in photon], dim=0) # (n_steps n_alives, 3)
        return result
    
    def flatten_rgbs(self, rgbs_list):
        return torch.cat([torch.reshape(rgbs, (-1,3)) for rgbs in rgbs_list], dim = 0)
    
    def extract_rgbs(self, photon_list):
        return [photon[...,6:9] for photon in photon_list]
    
    def clear(self):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        
        # self.fake_A_image_buffer.clear()
        # self.fake_B_image_buffer.clear()
        
        #* G_A(real_A)
        self.fake_B_rgbs, self.fake_B_image = None, None
        self.fake_B_photon = None
        #* G_B(fake_B)
        self.rec_A_rgbs,  self.rec_A_image  = None, None
        #* G_B(real_B) but G_B(fake_B)
        self.fake_A_rgbs, self.fake_A_image = None, None
        #* G_A(fake_A)
        self.rec_B_rgbs,  self.rec_B_image  = None, None        

        
        torch.cuda.empty_cache()  # GPU 캐시 비우기
        

    def load_photon_file(self, filename):
        H, W, cnt, n_actual_steps_list, indices_list, positions_list, directions_list, rgbs_list, densities_list, dists_list = load_binary_file(filename)

        image_data = volume_rendering(H, W, cnt, n_actual_steps_list, indices_list, positions_list, directions_list, rgbs_list, densities_list, dists_list)

        photon_data = [None] * cnt
        for i in range(cnt):
            indices = cp.repeat(indices_list[i][None, : , None], n_actual_steps_list[i+1], axis=0)
            photon_data[i] = cp.concatenate([positions_list[i], directions_list[i], rgbs_list[i], densities_list[i], dists_list[i], indices], axis=-1)
            photon_data[i] = torch.tensor(photon_data[i], dtype=torch.float32)
            photon_data[i] = photon_data[i].to(self.device) if not self.opt.data_parallel else photon_data[i].cuda()

        return H, W, photon_data, image_data.permute(2,1,0)
    
    def load_image_file(self, filename):
        image = read_image(filename) # 1 ~ 255의 torch.Tensor
        transform = transforms.Resize((800,800))
        return transform(image.type(torch.float32) / 255.)






if __name__ == "__main__":
    '''
    # 가능한 cuda number 출력
    cuda_d = torch.cuda.current_device()
    print(cuda_d)
    '''