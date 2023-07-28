import torch
from torch import nn

''' 
일단 지우기 
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        #! Label 미리 생성
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss() #* 더 좋다!
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']: #* 어려워요
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
'''

#! 다시 수정하기
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla'):

        super(GANLoss, self).__init__()
        #! Label 미리 생성
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss() #* 더 좋다!
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgan']: #* 어려워요
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        
    def forward(self, prediction, label):
        if self.gan_mode in ['lsgan', 'vanilla']:
            loss = self.loss(prediction, label)
        elif self.gan_mode == 'wgan':
            if label.dim() == 4:
                if label[0,0,0,0] == 1.0: loss = -prediction.mean()
                else: loss = prediction.mean()
            else:
                if label[0,0,0] == 1.0: loss = -prediction.mean()
                else: loss = prediction.mean()
        return loss
        



# 기존 GAN에서 사용되는 Loss와 동일
class AdversarialLoss(nn.Module):
    def __init__(self):
        '''
        두 개의  bce loss를 사용
        one for D(G(x)) vs real label (Generator)
        -> 최대한 진짜 같은 생성을 하기 위해 사용.

        one for D(y) vs real label and D(G(x)) vs fake label (Discriminator)
        -> generator는 discriminator가 진짜라고 착각하게 해야 한다.
        -> real 값은 real이라고 판별하게 해야 한다.
        '''
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()


    def foward_G(self, d_g_x, real):
        '''
        Generator에 대한 loss를 정의
        '''
        return self.loss(d_g_x, real)

    def forward_D(self, d_y, real, d_g_x, fake):
        '''
        Discriminator에 대한 loss를 정의
        '''
        d_real_loss = self.loss(d_y, real)
        d_fake_loss = self.loss(d_g_x, fake)

        result = (d_real_loss + d_fake_loss)/2

        return result


# CycleGAN의 주요 contribution Loss
class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.loss_forward = nn.L1Loss()
        self.loss_backward= nn.L1Loss()

    def forward(self, x, y, f_g_x, g_f_y):
        '''
        x, y -> true
        f_g_x, g_f_x -> predicted
        '''
        # forward와 backward 과정을 통해 2가지 domain에 대해서 비교가 가능해짐.
        loss_cyc = self.loss_forward(f_g_x, x) + self.loss_backward(g_f_y, y)

        return loss_cyc

# 이미지가 가진 고유한 정보를 잃지 않게 하기 위해서 사용(윤곽선 보존)
class IdentityLoss(nn.Module):

    def __init__(self):

        super(IdentityLoss, self).__init__()
        self.loss_identity_x = nn.L1Loss() # x domain에 대한 identity
        self.loss_identity_y = nn.L1Loss() # y domain에 대한 identity


    def forward(self, x, y, f_y, g_x):

        loss_iden = self.loss_identity_x(f_y, x) + self.loss_identity_y(g_x, y)

        return loss_iden
