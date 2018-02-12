from __future__ import absolute_import

from torchvision.transforms import *

import numpy as np
import torch


class ToPIL(object):
    def __call__(self, img):
        return Image.fromarray(img)


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


class RandomDistort(object):

    def __init__(self, kernel=3, std=1.0, probability=0.25):
        self.probability = probability
        self.kernel = kernel
        self.std = std

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        shape = img.shape
        # print(shape)
        dx = self.std * np.random.randn(shape[0], shape[1])
        dy = self.std * np.random.randn(shape[0], shape[1])
        # print(dx.shape)
        K = np.ones((self.kernel, self.kernel), dtype=np.float32) / (self.kernel* self.kernel)
        dxf = scipy.ndimage.filters.convolve(dx, K)
        dyf = scipy.ndimage.filters.convolve(dy, K)

        img_new = np.zeros_like(img, dtype=np.uint8)
        for h in range(shape[0]):
            for w in range(shape[1]):
                for c in range(shape[2]):
                    # print(w,h,c)
                    x = w + dyf[w, h]
                    y = h + dxf[w, h]
                    if x < 0 or y < 0 or x > shape[1]-1 or y > shape[0]-1:
                        img_new[h, w, c] = img[h, w, c]

                    else:
                        # print(x, y)
                        rx0 = max(min(int(x), shape[1]-1), 0)
                        rx1 = rx0 + 1
                        ry0 = max(min(int(y), shape[0]-1), 0)
                        ry1 = ry0 + 1
                        points = [(ry0, rx0, img[ry0,rx0,c]),
                                  (ry1, rx0, img[ry1,rx0,c]),
                                  (ry0, rx1, img[ry0,rx1,c]),
                                  (ry1, rx1, img[ry1,rx1,c])]
                        # print(points)
                        img_new[h, w, c] = int(bilinear_interpolation(y, x, points))
                        # print( img_new[h, w, c] )

        # plt.figure()
        # plt.imshow(dx, interpolation='none')
        # plt.savefig('d.png')
        # plt.figure()
        # plt.imshow(dxf, interpolation='none')
        # plt.savefig('df.png')

        # print(dxf.max())
        # print(dyf.max())

        # plt.imshow(img, interpolation='none')
        # plt.savefig('img.png')
        #
        # plt.imshow(img_new, interpolation='none')
        # plt.savefig('img_new.png')
        # raw_input('pause')

        return img_new

