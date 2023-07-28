from collections import deque
import random
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size # pool 크기
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        #! 학습되면서 더 질좋은 이미지가 계속 생겨남  (이미지가 수시로 갱신됨)
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class ImageReplayBuffer():

    def __init__(self, pool_size=100):
        self.buffer = deque(maxlen = pool_size)

    def append(self, x):
        self.buffer.append(x)

    def query(self, size=10):
        if len(self.buffer) < size:
            return self.deque2torch(len(self.buffer))
        else:
            return self.deque2torch(size)

    def deque2torch(self, size):
        images = random.sample(self.buffer, k=size)
        if len(images) == 1:
            return images[0]
        else:
            return torch.cat([image[None,...] for image in images], dim=0)



