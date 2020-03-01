import copy
import random
import torch
import scipy
from scipy.special import comb
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from skimage.morphology import label

import src.data.transforms

__all__ = [
    'compose', 'Compose', 'ToTensor', 'Normalize', 'RandomCrop',
    'RandomElasticDeformation', 'RandomHorizontalFlip', 'RandomVerticalFlip',
    'RandomRotation', 'NonLinearTransform', 'LocalPixelShuffling', 'Painting'
]


def compose(transforms=None):
    """Compose several transforms together.
    Args:
        transforms (Box): The preprocessing and augmentation techniques applied to the data.

    Returns:
        transforms (Compose): The list of BaseTransform.
    """
    if transforms is None:
        return None

    _transforms = []
    for transform in transforms:
        cls = getattr(src.data.transforms, transform.name)
        kwargs = transform.get('kwargs')
        _transforms.append(cls(**kwargs) if kwargs else cls())
    transforms = Compose(_transforms)
    return transforms


class BaseTransform:
    """The base class for all transforms.
    """

    def __init__(self):
        pass

    def __call__(self, *imgs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class Compose(BaseTransform):
    """Compose several transforms together.
    Args:
         transforms (Box): The preprocessing and augmentation techniques applied to the data.
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transformed.

        Returns:
            imgs (tuple of torch.Tensor): The transformed images.
        """
        for transform in self.transforms:
            imgs = transform(*imgs, **kwargs)

        # Returns the torch.Tensor instead of a tuple of torch.Tensor if there is only one image.
        if len(imgs) == 1:
            imgs = imgs[0]
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(BaseTransform):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *imgs, dtypes=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted to tensor.
            dtypes (sequence of torch.dtype, optional): The corresponding dtype of the images
                (default: None, transform all the images' dtype to torch.float).

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if dtypes:
            if not all(isinstance(dtype, torch.dtype) for dtype in dtypes):
                raise TypeError('All of the dtypes should be torch.dtype.')
            if len(dtypes) != len(imgs):
                raise ValueError('The number of the dtypes should be the same as the images.')
            imgs = tuple(img.to(dtype) for img, dtype in zip(map(torch.as_tensor, imgs), dtypes))
        else:
            imgs = tuple(img.float() for img in map(torch.as_tensor, imgs))
        return imgs


class Normalize(BaseTransform):
    """Normalize a tuple of images with the means and the standard deviations.
    Args:
        means (list, optional): A sequence of means for each channel (default: None).
        stds (list, optional): A sequence of standard deviations for each channel (default: None).
    """

    def __init__(self, means=None, stds=None):
        super().__init__()
        if means is None and stds is None:
            pass
        elif means is not None and stds is not None:
            if len(means) != len(stds):
                raise ValueError('The number of the means should be the same as the standard deviations.')
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

        self.means = means
        self.stds = stds

    def __call__(self, *imgs, normalize_tags=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be normalized.
            normalize_tags (sequence of bool, optional): The corresponding tags of the images
                (default: None, normalize all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The normalized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if normalize_tags:
            if len(normalize_tags) != len(imgs):
                raise ValueError('The number of the tags should be the same as the images.')
            if not all(normalize_tag in [True, False] for normalize_tag in normalize_tags):
                raise ValueError("All of the tags should be either True or False.")
        else:
            normalize_tags = [None] * len(imgs)

        _imgs = []
        for img, normalize_tag in zip(imgs, normalize_tags):
            if normalize_tag is None or normalize_tag is True:
                if self.means is None and self.stds is None:  # Apply image-level normalization.
                    axis = tuple(range(img.ndim - 1))
                    means = img.mean(axis=axis)
                    stds = img.std(axis=axis)
                    img = self._normalize(img, means, stds)
                else:
                    img = self._normalize(img, self.means, self.stds)
            elif normalize_tag is False:
                pass
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _normalize(img, means, stds):
        """Normalize the image with the means and the standard deviations.
        Args:
            img (numpy.ndarray): The image to be normalized.
            means (list): A sequence of means for each channel.
            stds (list): A sequence of standard deviations for each channel.

        Returns:
            img (numpy.ndarray): The normalized image.
        """
        img = img.copy()
        for c, mean, std in zip(range(img.shape[-1]), means, stds):
            img[..., c] = (img[..., c] - mean) / (std + 1e-10)
        return img


class RandomCrop(BaseTransform):
    """Crop a tuple of images at the same random location.
    Args:
        size (list): The desired output size of the cropped images.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be cropped.

        Returns:
            imgs (tuple of numpy.ndarray): The cropped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the cropped size should be the same as the image ({ndim - 1}). '
                             f'Got {len(self.size)}')

        if ndim == 3:
            h0, hn, w0, wn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0:hn, w0:wn] for img in imgs])
        elif ndim == 4:
            h0, hn, w0, wn, d0, dn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0:hn, w0:wn, d0:dn] for img in imgs])
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (numpy.ndarray): The image to be cropped.
            size (list): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        if any(i - j < 0 for i, j in zip(img.shape, size)):
            raise ValueError(f'The image ({img.shape}) is smaller than the cropped size ({size}). '
                             'Please use a smaller cropped size.')

        if img.ndim == 3:
            h, w = img.shape[:-1]
            ht, wt = size
            h0, w0 = random.randint(0, h - ht), random.randint(0, w - wt)
            return h0, h0 + ht, w0, w0 + wt
        elif img.ndim == 4:
            h, w, d = img.shape[:-1]
            ht, wt, dt = size
            h0, w0, d0 = random.randint(0, h - ht), random.randint(0, w - wt), random.randint(0, d - dt)
            return h0, h0 + ht, w0, w0 + wt, d0, d0 + dt


class RandomElasticDeformation(BaseTransform):
    """Do the random elastic deformation as used in U-Net and V-Net by using the bspline transform.
    Args:
        do_z_deformation (bool, optional): Whether to apply the deformation along the z dimension (default: False).
        num_ctrl_points (int, optional): The number of the control points to form the control point grid (default: 4).
        sigma (int or float, optional): The number to determine the extent of deformation (default: 15).
        prob (float, optional): The probability of applying the deformation (default: 0.5).
    """

    def __init__(self, do_z_deformation=False, num_ctrl_points=4, sigma=15, prob=0.5):
        super().__init__()
        self.do_z_deformation = do_z_deformation
        self.num_ctrl_points = max(num_ctrl_points, 2)
        self.sigma = max(sigma, 1)
        self.prob = max(0, min(prob, 1))
        self.bspline_transform = None

    def __call__(self, *imgs, elastic_deformation_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            elastic_deformation_orders (sequence of int, optional): The corresponding interpolation order of the images
                (default: None, the interpolation order would be 3 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The deformed images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            self._init_bspline_transform(imgs[0].shape)
            if elastic_deformation_orders:
                imgs = tuple(self._apply_bspline_transform(img, order)
                             for img, order in zip(imgs, elastic_deformation_orders))
            else:
                imgs = map(self._apply_bspline_transform, imgs)
        return imgs

    def _init_bspline_transform(self, shape):
        """Initialize the bspline transform.
        Args:
            shape (tuple): The size of the control point grid.
        """
        # Remove the channel dimension.
        shape = shape[:-1]

        # Initialize the control point grid.
        img = sitk.GetImageFromArray(np.zeros(shape))
        mesh_size = [self.num_ctrl_points] * img.GetDimension()
        self.bspline_transform = sitk.BSplineTransformInitializer(img, mesh_size)

        # Set the parameters of the bspline transform randomly.
        params = self.bspline_transform.GetParameters()
        params = np.asarray(params, dtype=np.float64)
        params = params + np.array(list(random.gauss(0, self.sigma) for _ in range(params.shape[0])))
        if len(shape) == 3 and not self.do_z_deformation:
            params[0: len(params) // 3] = 0
        params = tuple(params)
        self.bspline_transform.SetParameters(params)

    def _apply_bspline_transform(self, img, order=3):
        """Apply the bspline transform.
        Args:
            img (np.ndarray): The image to be deformed.
            order (int, optional): The interpolation order (default: 3, should be 0, 1 or 3).

        Returns:
            img (np.ndarray): The deformed image.
        """
        # Create the resampler.
        resampler = sitk.ResampleImageFilter()
        if order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError(f'The interpolation order should be 0, 1 or 3. Got {order}.')

        # Apply the bspline transform.
        shape = img.shape
        img = sitk.GetImageFromArray(np.squeeze(img))
        resampler.SetReferenceImage(img)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self.bspline_transform)
        img = resampler.Execute(img)
        img = sitk.GetArrayFromImage(img).reshape(shape)
        return img


class RandomHorizontalFlip(BaseTransform):
    """Do the random flip horizontally.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.

        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([np.flip(img, 1).copy() for img in imgs])
        return imgs


class RandomVerticalFlip(BaseTransform):
    """Do the random flip vertically.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.

        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([np.flip(img, 0).copy() for img in imgs])
        return imgs


class RandomRotation(BaseTransform):
    """
    """

    def __init__(self, prob=0.5, min_angle=None, max_angle=None, rotation_angle=None, spatial_rank=3):
        super().__init__()
        self.prob = max(0, min(prob, 1))
        self._transform = None
        
        assert (min_angle is not None and max_angle is not None) or rotation_angle is not None, \
                "Need to identify the rotation angles"
        if rotation_angle is not None:
            self.init_non_uniform_angle(*rotation_angle)
        else:
            self.init_uniform_angle([min_angle, max_angle])
        self.randomise(spatial_rank)
        
    def init_uniform_angle(self, rotation_angle):
        assert rotation_angle[0] < rotation_angle[1]
        self.min_angle = float(rotation_angle[0])
        self.max_angle = float(rotation_angle[1])

    def init_non_uniform_angle(self, rotation_angle_x, rotation_angle_y, rotation_angle_z=None):
        if len(rotation_angle_x):
            assert rotation_angle_x[0] < rotation_angle_x[1]
        if len(rotation_angle_y):
            assert rotation_angle_y[0] < rotation_angle_y[1]
        self.rotation_angle_x = [float(e) for e in rotation_angle_x]
        self.rotation_angle_y = [float(e) for e in rotation_angle_y]
        
        if rotation_angle_z is not None:
            if len(rotation_angle_z):
                assert rotation_angle_z[0] < rotation_angle_z[1]
            self.rotation_angle_z = [float(e) for e in rotation_angle_z]

    def randomise(self, spatial_rank=3):
        self._randomise_transformation(spatial_rank)

    def _randomise_transformation(self, spatial_rank):
        angle_x = 0.0
        angle_y = 0.0
        angle_z = 0.0
        if self.min_angle is None and self.max_angle is None:
            # generate transformation
            if len(self.rotation_angle_x) >= 2:
                angle_x = np.random.uniform(
                    self.rotation_angle_x[0],
                    self.rotation_angle_x[1]) * np.pi / 180.0

            if len(self.rotation_angle_y) >= 2:
                angle_y = np.random.uniform(
                    self.rotation_angle_y[0],
                    self.rotation_angle_y[1]) * np.pi / 180.0

            if (spatial_rank == 3) and len(self.rotation_angle_z) >= 2:
                angle_z = np.random.uniform(
                    self.rotation_angle_z[0],
                    self.rotation_angle_z[1]) * np.pi / 180.0
        else:
            # generate transformation
            angle_x = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
            angle_y = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
            if (spatial_rank == 3):
                angle_z = np.random.uniform(
                    self.min_angle, self.max_angle) * np.pi / 180.0

        transform_x = np.array([[np.cos(angle_x), -np.sin(angle_x), 0.0],
                                [np.sin(angle_x), np.cos(angle_x), 0.0],
                                [0.0, 0.0, 1.0]])
        transform_y = np.array([[np.cos(angle_y), 0.0, np.sin(angle_y)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angle_y), 0.0, np.cos(angle_y)]])
        if (spatial_rank == 3):
            transform_z = np.array([[1.0, 0.0, 0.0],
                                    [0.0, np.cos(angle_z), -np.sin(angle_z)],
                                    [0.0, np.sin(angle_z), np.cos(angle_z)]])
            transform = np.dot(transform_z, np.dot(transform_x, transform_y))
        else:
            transform = np.dot(transform_x, transform_y)
        self._transform = transform

    def _apply_transformation(self, image, order=3):
        if order < 0:
            return image
        assert self._transform is not None
        
        for c in range(image.shape[-1]):
            center_ = 0.5 * np.asarray(image.shape[:-1], dtype=np.int64)
            c_offset = center_ - center_.dot(self._transform)
            image[..., c] = scipy.ndimage.affine_transform(
                image[..., c], self._transform.T, c_offset, order=order)
        return image

    def __call__(self, *imgs, rotation_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            rotation_orders (sequence of int, optional): The corresponding interpolation order of the images
                (default: None, the interpolation order would be 3 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The deformed images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            if rotation_orders:
                imgs = tuple(self._apply_transformation(img, order)
                             for img, order in zip(imgs, rotation_orders))
            else:
                imgs = map(self._apply_transformation, imgs)
        return imgs
    

class NonLinearTransform(BaseTransform):
    """
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transform.

        Returns:
            imgs (tuple of numpy.ndarray): The transformed images.
        """
        if len(imgs) != 1:
            raise ValueError('The transform only supports single image in the current version.')
            
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([self._apply_transformation(imgs[0])])
        return imgs

    def _apply_transformation(self, img):
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
            
        transformed_img = np.interp(img, xvals, yvals)
        return transformed_img
    
    @classmethod
    def bezier_curve(cls, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.
           Control points should be a list of lists, or list of tuples
           such as [[1,1], 
                    [2,3], 
                    [4,5], ... [Xn, Yn]]
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """
        num_points = len(points)
        xpoints = np.array([p[0] for p in points])
        ypoints = np.array([p[1] for p in points])
        
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array([cls.bernstein_poly(i, num_points-1, t) for i in range(0, num_points)])        
        xvals = np.dot(xpoints, polynomial_array)
        yvals = np.dot(ypoints, polynomial_array)
        return xvals, yvals
    
    @classmethod
    def bernstein_poly(cls, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * (t ** (n-i)) * ((1 - t) ** i)


class LocalPixelShuffling(BaseTransform):
    """
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transform.

        Returns:
            imgs (tuple of numpy.ndarray): The transformed images.
        """
        if len(imgs) != 1:
            raise ValueError('The transform only supports single image in the current version.')
        
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
            
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([self.local_pixel_shuffling(imgs[0])])
        return imgs

    @staticmethod
    def local_pixel_shuffling(img):
        """
        Args:
            img (numpy.ndarray, ndim=3or4): The images to be transform.

        Returns:
            imgs (numpy.ndarray): The transformed images.
        """
        num_block = 10000
        original_img = copy.deepcopy(img)
        transformed_img = copy.deepcopy(img)
        
        if img.ndim == 3:
            H, W, C = img.shape
            for _ in range(num_block):
                block_noise_size_x = random.randint(1, H//10)
                block_noise_size_y = random.randint(1, W//10)
                noise_x = random.randint(0, H-block_noise_size_x)
                noise_y = random.randint(0, W-block_noise_size_y)
                window = original_img[noise_x:noise_x+block_noise_size_x,
                                      noise_y:noise_y+block_noise_size_y]
                window = window.flatten()
                np.random.shuffle(window)
                window = window.reshape((block_noise_size_x, 
                                         block_noise_size_y,
                                         C))
                transformed_img[noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y] = window
        elif img.ndim == 4:
            H, W, D, C = img.shape
            for _ in range(num_block):
                block_noise_size_x = random.randint(1, H//10)
                block_noise_size_y = random.randint(1, W//10)
                block_noise_size_z = random.randint(1, D//10)
                noise_x = random.randint(0, H-block_noise_size_x)
                noise_y = random.randint(0, W-block_noise_size_y)
                noise_z = random.randint(0, D-block_noise_size_z)
                window = original_img[noise_x:noise_x+block_noise_size_x, 
                                      noise_y:noise_y+block_noise_size_y, 
                                      noise_z:noise_z+block_noise_size_z]
                window = window.flatten()
                np.random.shuffle(window)
                window = window.reshape((block_noise_size_x, 
                                         block_noise_size_y, 
                                         block_noise_size_z,
                                         C))
                transformed_img[noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y, 
                                noise_z:noise_z+block_noise_size_z] = window
        return transformed_img


class Painting(BaseTransform):
    """
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
        inpaint_rate (float, optional): The rate of applying the in-painting. The rate of out-painting is `1 - inpaint_rate`. (default: 0.2).
    """
    def __init__(self, prob=0.5, inpaint_rate=0.2):
        self.prob = max(0, min(prob, 1))
        self.inpaint_rate = max(0, min(inpaint_rate, 1))
        self.outpaint_rate = 1.0 - self.inpaint_rate

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transform.

        Returns:
            imgs (tuple of numpy.ndarray): The transformed images.
        """
        if len(imgs) != 1:
            raise ValueError('The transform only supports single image in the current version.')
        
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
            
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")
            
        if random.random() < self.prob:
            if random.random() < self.inpaint_rate:
                imgs = tuple([self.in_painting(imgs[0])])
            else:
                imgs = tuple([self.out_painting(imgs[0])])
        return imgs

    @staticmethod
    def in_painting(img):
        num_painting = 5
        img = copy.deepcopy(img)
        if img.ndim == 3:
            H, W, C = img.shape
            while num_painting > 0:
                block_noise_size_x = random.randint(H//6, H//3)
                block_noise_size_y = random.randint(W//6, W//3)
                noise_x = random.randint(3, H-block_noise_size_x-3)
                noise_y = random.randint(3, W-block_noise_size_y-3)
                img[noise_x:noise_x+block_noise_size_x, 
                    noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x, 
                                                                         block_noise_size_y,
                                                                         C) * 1.0
                num_painting -= 1
        elif img.ndim == 4:
            H, W, D, C = img.shape
            while num_painting > 0:
                block_noise_size_x = random.randint(H//6, H//3)
                block_noise_size_y = random.randint(W//6, W//3)
                block_noise_size_z = random.randint(D//6, D//3)
                noise_x = random.randint(3, H-block_noise_size_x-3)
                noise_y = random.randint(3, W-block_noise_size_y-3)
                noise_z = random.randint(3, D-block_noise_size_z-3)
                img[noise_x:noise_x+block_noise_size_x, 
                    noise_y:noise_y+block_noise_size_y, 
                    noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                                         block_noise_size_y, 
                                                                         block_noise_size_z,
                                                                         C) * 1.0
                num_painting -= 1
        return img

    @staticmethod
    def out_painting(img):
        num_painting = 5
        img = copy.deepcopy(img)
        tmp_img = copy.deepcopy(img)
        if img.ndim == 3:
            H, W, C = img.shape
            img = np.random.rand(H, W, C) * 1.0
            while num_painting > 0:
                block_noise_size_x = H - random.randint(3*H//7, 4*H//7)
                block_noise_size_y = W - random.randint(3*W//7, 4*W//7)
                noise_x = random.randint(3, H-block_noise_size_x-3)
                noise_y = random.randint(3, W-block_noise_size_y-3)
                img[noise_x:noise_x+block_noise_size_x, 
                    noise_y:noise_y+block_noise_size_y] = tmp_img[noise_x:noise_x+block_noise_size_x,
                                                                  noise_y:noise_y+block_noise_size_y]
                num_painting -= 1
        elif img.ndim == 4:
            H, W, D, C = img.shape
            img = np.random.rand(H, W, D, C) * 1.0
            while num_painting > 0:
                block_noise_size_x = H - random.randint(3*H//7, 4*H//7)
                block_noise_size_y = W - random.randint(3*W//7, 4*W//7)
                block_noise_size_z = D - random.randint(3*D//7, 4*D//7)
                noise_x = random.randint(3, H-block_noise_size_x-3)
                noise_y = random.randint(3, W-block_noise_size_y-3)
                noise_z = random.randint(3, D-block_noise_size_z-3)
                img[noise_x:noise_x+block_noise_size_x, 
                    noise_y:noise_y+block_noise_size_y, 
                    noise_z:noise_z+block_noise_size_z] = tmp_img[noise_x:noise_x+block_noise_size_x,
                                                                  noise_y:noise_y+block_noise_size_y,
                                                                  noise_z:noise_z+block_noise_size_z]
                num_painting -= 1
        return img