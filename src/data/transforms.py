import random
import torch
import numpy as np
import SimpleITK as sitk
from collections.abc import Iterable

import src.data.transforms

__all__ = [
    'Compose',
    'ToTensor',
    'Normalize',
    'RandomCrop',
    'RandomElasticDeformation',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
]


class Compose:
    """Compose several transforms together.
    Args:
         transforms (sequence of BaseTransform, optional): The preprocessing and augmentation techniques 
            applied to the data (default: None, do not apply any transform).
    """

    def __init__(self, transforms=None):
        if transforms is None:
            self.transforms = tuple()
        else:
            if not isinstance(transforms, Iterable):
                transforms = tuple(transforms)
            if not all(isinstance(transform, BaseTransform) for transform in transforms):
                raise TypeError('All of the transforms should be BaseTransform.')
            self.transforms = tuple(transforms)

    def __call__(self, *imgs, **transform_kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transformed.
            transform_kwargs (dict): The runtime kwargs for each transforms.

        Returns:
            imgs (tuple): The transformed images.
        """
        for transform in self.transforms:
            kwargs = transform_kwargs.get(transform.__class__.__name__, {})
            tags = kwargs.get('tags', tuple(True for _ in range(len(imgs))))
            if len(tags) != len(imgs):
                raise ValueError('The number of the tags should be the same as the images.')
            if not all(tag in [True, False] for tag in tags):
                raise ValueError("All of the tags should be either True or False.")
            if not any(tags):
                raise ValueError("The tags should not be all False.")

            if all(tags):
                imgs = transform(*imgs, **kwargs)
            else:
                _imgs = tuple(img for img, tag in zip(imgs, tags) if tag)
                _imgs = transform(*_imgs, **kwargs)
                imgs = np.array(imgs)
                imgs[np.array(tags)] = _imgs
                imgs = tuple(imgs)
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

    @classmethod
    def compose(cls, transforms=None):
        """Compose several transforms together.
        Args:
            transforms (Box, optional): The preprocessing and augmentation techniques 
                applied to the data (default: None, do not apply any transform).

        Returns:
            instance (Compose): The Compose instance.
        """
        if transforms is None:
            return cls(tuple())
        if not isinstance(transforms, Box):
            raise ValueError('The type of the transforms should be Box.')

        _transforms = tuple()
        for transform in transforms:
            transform_cls = getattr(src.data.transforms, transform.name)
            _transforms.add(transform_cls(**config.get('kwargs', {})))
        return cls(_transforms)


class BaseTransform:
    """The base class for all transforms.
    """

    def __init__(self):
        pass

    def __call__(self, *imgs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class ToTensor(BaseTransform):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    Default is to transform a tuple of numpy.ndarray to a tuple of torch.FloatTensor.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *imgs, dtypes=None):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted.
            dtypes (sequence of torch.dtype, optional): The dtypes of the converted images
                (default: None, transform all the images' dtype to torch.float).

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if dtypes is None:
            imgs = tuple(img.float() for img in map(torch.as_tensor, imgs))
        else:
            if not isinstance(dtypes, Iterable):
                dtypes = tuple(dtypes)
            if not all(isinstance(dtype, torch.dtype) for dtype in dtypes):
                raise TypeError('All of the dtypes should be torch.dtype.')
            if len(dtypes) != len(imgs):
                raise ValueError('The number of the dtypes should be the same as the images.')
            imgs = tuple(img.to(dtype) for img, dtype in zip(map(torch.as_tensor, imgs), dtypes))
        return imgs


class Normalize(BaseTransform):
    """Normalize a tuple of images with the means and the standard deviations.
    Default is to apply image-level normalization to zero-mean and unit-variance per channel.

    Args:
        means (scalar or sequence, optional): The means for each channel (default: None).
        stds (scalar or sequence, optional): The standard deviations for each channel (default: None).
        per_channel (bool): Whether to apply image-level normalization per channel (default: True).
            Note that this argument is only valid when means and stds are both None.
    """

    def __init__(self, means=None, stds=None, per_channel=True):
        super().__init__()
        if means is None and stds is None:
            self.means = None
            self.stds = None
            self.per_channel = per_channel
        elif means is not None and stds is not None:
            self.means = np.array(means)
            self.stds = np.array(stds)
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

    def __call__(self, *imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be normalized.

        Returns:
            imgs (tuple of numpy.ndarray): The normalized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        _imgs = tuple()
        for img in imgs:
            if self.means is None and self.stds is None:  # Apply image-level normalization.
                axis = tuple(range(img.ndim - 1)) if self.per_channel else tuple(range(img.ndim))
                means = img.mean(axis=axis)
                stds = img.std(axis=axis)
                img = self._normalize(img, means, stds)
            else:
                img = self._normalize(img, self.means, self.stds)
            _imgs.add(img)
        imgs = _imgs
        return imgs

    @staticmethod
    def _normalize(img, means, stds):
        """Normalize the image with the means and the standard deviations.
        Args:
            img (numpy.ndarray): The image to be normalized.
            means (numpy.ndarray): The means for each channel.
            stds (numpy.ndarray): The standard deviations for each channel.

        Returns:
            img (numpy.ndarray): The normalized image.
        """
        img = img.copy()
        img = (img - means) / stds.clip(min=1e-10)
        return img


class RandomCrop(BaseTransform):
    """Crop a tuple of images at the same random location.
    Args:
        size (sequence): The desired output size of the cropped images.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, *imgs):
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
            raise ValueError(f'The dimensions of the cropped size should be the same as '
                             f'the image ({ndim - 1}). Got {len(self.size)}.')

        if ndim == 3:
            h0, hn, w0, wn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple(img[h0:hn, w0:wn] for img in imgs)
        elif ndim == 4:
            h0, hn, w0, wn, d0, dn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple(img[h0:hn, w0:wn, d0:dn] for img in imgs)
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (numpy.ndarray): The image to be cropped.
            size (sequence): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        if any(i < j for i, j in zip(img.shape[:-1], size)):
            raise ValueError(f'The image size {img.shape[:-1]} is smaller than '
                             f'the cropped size {size}. Please use a smaller cropped size.')

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
    Ref: 
        https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/rand_elastic_deform.html

    Args:
        do_z_deformation (bool, optional): Whether to apply the deformation
            along the z dimension (default: False).
        num_ctrl_points (int, optional): The number of the control points
            to form the control point grid (default: 4).
        sigma (scalar, optional): The number to determine 
            the extent of deformation (default: 15).
        prob (float, optional): The probability of applying the deformation (default: 0.5).
    """

    def __init__(self, do_z_deformation=False, num_ctrl_points=4, sigma=15, prob=0.5):
        super().__init__()
        self.do_z_deformation = do_z_deformation
        self.num_ctrl_points = max(num_ctrl_points, 2)
        self.sigma = max(sigma, 1)
        self.prob = max(0, min(prob, 1))
        self.bspline_transform = None

    def __call__(self, *imgs, orders=None):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            orders (sequence of int, optional): The corresponding interpolation order of the images
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
        params = params + np.array(tuple(random.gauss(0, self.sigma) for _ in range(params.shape[0])))
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

    def __call__(self, *imgs):
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
            imgs = tuple(np.flip(img, 1) for img in imgs)
        return imgs


class RandomVerticalFlip(BaseTransform):
    """Do the random flip vertically.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs):
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
            imgs = tuple(np.flip(img, 0) for img in imgs)
        return imgs
