import torch

from torch import nn
from collections import OrderedDict
from typing import Union


class ResidualBlock(nn.Module):
  def __init__(self, in_channels:int, out_channels:int):
    super(ResidualBlock, self).__init__() # 상속받은 nn.Module의 생성자 호출

    # in_channels와 out_channels를 update
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.block = nn.Sequential(
        nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3,
                  padding='same', padding_mode='reflect'),
                  # padding='same'은 입력데이터와 출력데이터의 크기가 동일하다는 뜻
        nn.InstanceNorm2d(self.out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3,
                  padding='same', padding_mode='reflect'),
        nn.InstanceNorm2d(self.out_channels)
        )
    def forward(self, x):
      output = self.block(x) + x

      return output


class GeneratorMLP(nn.Module):
  def __init__(self, in_features, out_features):
    super(GeneratorMLP, self).__init__()
    self.in_features = in_features
    self.out_features = out_features

    #! 32,48은 가능
    #! 64는 불가능 (다만, identity loss 제외하면 가능)
    self.linear1 = nn.Linear(self.in_features, 64)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(64, self.out_features)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.sigmoid(x)
    return x






class Generator(nn.Module):
  def __init__(self, init_channels:int, kernel_size:int, stride:int, n_blocks:int=6):
    super(Generator, self).__init__() # nn.Module을 상속받음

    self.init_channels=init_channels
    self.kernel_size=kernel_size
    self.stride=stride
    self.n_blocks=n_blocks

    layers = OrderedDict()
    # 첫번째 layer를 쌓는 과정
    layers['conv_first'] = self._make_block(in_channels=3, out_channels=self.init_channels,
                                            kernel_size=7, stride=1, padding='same')
    # downsampling을 위한 convolution layers
    # init_channels = 64
    # 0 ~ 1까지의 값
    for i in range(2):
      ic = self.init_channels*(i+1)
      k = 2 * ic
      layers[f'd_{k}'] = self._make_block(in_channels=ic, out_channels=k,
                                          kernel_size=self.kernel_size, stride=stride)

    # Create Residual Block
    # n_blocks = 6, k = 128
    for i in range(self.n_blocks):
      layers[f'R{k}_{i+1}'] = ResidualBlock(k ,k)

    # Upsampling을 위한 convolution layers
    # k = 128
    for i in range(2):
      k = int(k/2)
      layers[f'u_{k}'] = self._make_block(in_channels=k*2, out_channels=k, kernel_size=self.kernel_size,
                                          stride=self.stride, mode='u')

    # last_conv layer
    layers['conv_last'] = nn.Conv2d(in_channels=self.init_channels, out_channels=3, kernel_size=7,
                                    stride=1, padding='same', padding_mode='reflect') # stride 왜 갑자기 1????
    layers['tanh'] = nn.Tanh() # tanh는 왜 갑자기 진행하는 것인지?????

    self.model = nn.Sequential(
        layers
    )

  def forward(self, x):

    op = self.model(x) # 최종 모델 return

    return op

  #  convolution block을 생성하는 함수 작성
  # 이 과정을 따로 분류하여 보다 직관적으로 block을 쌓는 과정을 이해할 수 있게 되었다.
  def _make_block(self, in_channels:int, out_channels:int, kernel_size:int,
                  stride:int, padding:Union[int, str]=1, mode:str='d'):

    # Union은 여러개의 type들을 결합하는 데에 사용하는 특수한 generic한 타입이다.
    """
    in_channels (int형) : input feature map의 channels
    out_channels (int형) : output feature map의 channels
    kernel_size (int형) : kernel window size
    stride (int형) : convolution의 stride
    padding (int형) : padding을 진행할 양에 대한 정보를 정함
    mode (str형) : 'd' downsampling mode(default), 'u' upsampling mode
    """

    # U-Net형식을 사용하기 위해 Downsampling & Upsampling을 사용하였다.
    block = []
    if mode.lower() == 'd':
      block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, padding_mode='reflect'))

    elif mode.lower() == 'u': # ConvTranspose2d 함수는 upsampling을 진행할 때,
                              # 사용되는 함수로 Transpose 개념이 들어있다.
      block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                     padding=padding, output_padding=1))

    # 실질적으로 block을 쌓는 과정
    '''
    - out_channels를 이용하여 instance의 정규화를 진행한다.
    - 추가적으로 ReLU function을 활용하여 block을 활성화한다.
    '''
    block += [nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]

    # block을 nn.Sequential에 쌓아서 출력
    return nn.Sequential(*block)



class Discriminator(nn.Module):
  def __init__(self, n_layers:int=6, input_c:int=3, n_filter:int=64, kernel_size:int=9):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential()
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    layers = []

    # Layer 개수만큼 block을 쌓아간다.
    for i in range(self.n_layers):
      if i == 0:
        ic, oc = input_c, n_filter
        layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2, padding=1, normalize=False))
      else:
        ic = oc # 이전에 진행한 block이 존재하기 때문이다.
        '''
        discriminator에서 channel 수가 증가하는 이유는 이미지의 진위여부를 판단하기 위한 방법으로 다양한 특징들을 고려하기 위해서이다.
        #! 다른 데이터셋에서 사용될 경우 바뀔 수도 있다.
        '''
        oc = 2*ic
        stride = 2

        if i == self.n_layers-1:
          stride=1

        layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=stride, padding=1))

    #! 최종결과값을 확인하기 위해 output channel을 1로 설정
    layers.append(nn.Conv2d(oc, 1, stride=stride, kernel_size=self.kernel_size, padding=1))

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

  # Dircriminator는 Gennerator처럼 U-Net이 아닌 Conv2d만을 사용한다.
  def _make_block(self, in_channels, out_channels, stride, kernel_size=3, padding=0, normalize=True):
    layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding)]
    if normalize:
      layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True)) # 여기선 왜 ReLU가 아닌 LeakyReLU를 쓰는지 모르겠음.

    return nn.Sequential(*layers)
