import functools
import random
import torch
import numpy as np
import SimpleITK as sitk
from box import BoxList

import src.data.transforms

__all__ = [
    'Compose',
    'ToTensor',
    'Normalize',
    'MinMaxScale',
    'Clip',
    'Resample',
    'RandomCrop',
    'RandomElasticDeform',
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
            if not all(isinstance(transform, BaseTransform) for transform in transforms):
                raise TypeError('All of the transforms should be BaseTransform.')
            self.transforms = tuple(transforms)

    def __call__(self, *imgs, **transforms_kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transformed.
            transforms_kwargs (dict): The runtime kwargs for each transforms.

        Returns:
            imgs (tuple): The transformed images.
        """
        for transform in self.transforms:
            kwargs = transforms_kwargs.get(transform.__class__.__name__, {})
            transformed = kwargs.pop('transformed', tuple(True for _ in range(len(imgs))))
            if len(transformed) != len(imgs):
                raise ValueError('The number of the transformed should be the same as the images.')
            if not all(_transformed in [True, False] for _transformed in transformed):
                raise ValueError('All of the transformed should be either True or False.')
            if not any(transformed):
                raise ValueError('The transformed should not be all False.')

            if all(transformed):
                imgs = transform(*imgs, **kwargs)
            else:
                transformed_imgs, untransformed_imgs = self._split_imgs(imgs, transformed)
                transformed_imgs = transform(*transformed_imgs, **kwargs)
                imgs = self._reassemble_imgs(transformed_imgs, untransformed_imgs, transformed)
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n'
            format_string += f'    {transform}'
        format_string += '\n)'
        return format_string

    @classmethod
    def compose(cls, transforms=None):
        """Compose several transforms together.
        Args:
            transforms (BoxList, optional): The preprocessing and augmentation techniques
                applied to the data (default: None, do not apply any transform).

        Returns:
            instance (Compose): The Compose instance.
        """
        if transforms is None:
            return cls(tuple())
        if not isinstance(transforms, BoxList):
            raise ValueError('The type of the transforms should be BoxList.')

        _transforms = []
        for transform in transforms:
            transform_cls = getattr(src.data.transforms, transform.name)
            _transforms.append(transform_cls(**transform.get('kwargs', {})))
        return cls(tuple(_transforms))

    @staticmethod
    def _split_imgs(imgs, transformed):
        """Split the images into transformed and untransformed ones by condition.
        Args:
            imgs (tuple of numpy.ndarray): The images to be splited.
            transformed (tuple of bool): Specify which image should be transformed.

        Returns:
            transformed_imgs (tuple of numpy.ndarray): The images should be transformed.
            untransformed_imgs (tuple of numpy.ndarray): The images should not be transformed.
        """
        transformed_imgs, untransformed_imgs = [], []
        for img, _transformed in zip(imgs, transformed):
            (transformed_imgs if _transformed else untransformed_imgs).append(img)
        return tuple(transformed_imgs), tuple(untransformed_imgs)

    @staticmethod
    def _reassemble_imgs(transformed_imgs, untransformed_imgs, transformed):
        """Reassemble the images from transformed and untransformed ones by condition.
        Args:
            transformed_imgs (tuple of numpy.ndarray): The images have been transformed.
            untransformed_imgs (tuple of numpy.ndarray): The images have not been transformed.
            transformed (tuple of bool): Specify which image has been transformed.

        Returns:
            imgs (tuple of numpy.ndarray): The reassembled images.
        """
        transformed_imgs_iterator = iter(transformed_imgs)
        untransformed_imgs_iterator = iter(untransformed_imgs)
        imgs = tuple(
            next(transformed_imgs_iterator) if _transformed else next(untransformed_imgs_iterator)
            for _transformed in transformed
        )
        return imgs


class BaseTransform:
    """The base class for all transforms.
    """

    def __call__(self, *imgs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class ToTensor(BaseTransform):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    Default is to transform a tuple of numpy.ndarray to a tuple of channel-first torch.Tensor
    which infer data types from the tuple of numpy.ndarray.

    Args:
        channel_first (bool, optional): Whether the data format of the output data is channel-first,
            that is (H, W, C) to (C, H, W) and (H, W, D, C) to (C, D, H, W) (default: True).
    """

    def __init__(self, channel_first=True):
        super().__init__()
        self.channel_first = channel_first

    def __call__(self, *imgs, dtypes=None):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted.
            dtypes (sequence of torch.dtype, optional): The dtypes of the converted images
                (default: None, infer data types from the images).

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError('All of the images should be 2D or 3D with channels.')

        if dtypes is None:
            dtypes = tuple(None for _ in range(len(imgs)))
        if len(dtypes) != len(imgs):
            raise ValueError('The number of the dtypes should be the same as the images.')
        imgs = tuple(torch.as_tensor(img, dtype=dtype) for img, dtype in zip(imgs, dtypes))
        if self.channel_first:
            if imgs[0].ndim == 3:
                imgs = tuple(img.permute(2, 0, 1).contiguous() for img in imgs)
            elif imgs[0].ndim == 4:
                imgs = tuple(img.permute(3, 2, 0, 1).contiguous() for img in imgs)
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
            raise ValueError('All of the images should be 2D or 3D with channels.')

        _imgs = []
        for img in imgs:
            if self.means is None and self.stds is None:  # Apply image-level normalization.
                axis = tuple(range(img.ndim - 1)) if self.per_channel else tuple(range(img.ndim))
                means = img.mean(axis=axis)
                stds = img.std(axis=axis)
                img = self._normalize(img, means, stds)
            else:
                img = self._normalize(img,
                                      self.means.astype(img.dtype),
                                      self.stds.astype(img.dtype))
            _imgs.append(img)
        imgs = tuple(_imgs)
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


class MinMaxScale(BaseTransform):
    """Scale a tuple of images to a given range with the minimum and maximum values.
    Default is to apply image-level scaling to [0, 1] per channel.

    Args:
        mins (scalar or sequence, optional): The minimum values for each channel (default: None).
        maxs (scalar or sequence, optional): The maximum values for each channel (default: None).
        per_channel (bool): Whether to apply image-level scaling per channel (default: True).
            Note that this argument is only valid when mins and maxs are both None.
        value_range (sequence, optional): The minimum and maximum value after scaling.
    """

    def __init__(self, mins=None, maxs=None, per_channel=True, value_range=(0, 1)):
        super().__init__()
        if mins is None and maxs is None:
            self.mins = None
            self.maxs = None
            self.per_channel = per_channel
        elif mins is not None and maxs is not None:
            self.mins = np.array(mins)
            self.maxs = np.array(maxs)
        else:
            raise ValueError('Both the mins and the maxs should have values or None.')

        min_, max_ = value_range
        if min_ > max_:
            raise ValueError('The minimum value of value_range should be smaller than the maximum value. '
                             f'Got {value_range}.')
        self.value_range = np.array(value_range)

    def __call__(self, *imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be scaled.

        Returns:
            imgs (tuple of numpy.ndarray): The scaled images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError('All of the images should be 2D or 3D with channels.')

        _imgs = []
        for img in imgs:
            if self.mins is None and self.maxs is None:  # Apply image-level scaling.
                axis = tuple(range(img.ndim - 1)) if self.per_channel else tuple(range(img.ndim))
                mins = img.min(axis=axis)
                maxs = img.max(axis=axis)
                img = self._min_max_scale(img, mins, maxs, self.value_range.astype(img.dtype))
            else:
                img = self._min_max_scale(img,
                                          self.mins.astype(img.dtype),
                                          self.maxs.astype(img.dtype),
                                          self.value_range.astype(img.dtype))
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _min_max_scale(img, mins, maxs, value_range):
        """Scale the image with the minimum and maximum values.
        Args:
            img (numpy.ndarray): The image to be scaled.
            mins (numpy.ndarray): The minimum values for each channel.
            maxs (numpy.ndarray): The maximum values for each channel.
            value_range (numpy.ndarray): The minimum and minimum value after scaling.

        Returns:
            img (numpy.ndarray): The scaled image.
        """
        img = img.copy()
        img = (img - mins) / (maxs - mins).clip(min=1e-10)
        min_, max_ = value_range
        img = img * (max_ - min_) + min_
        return img


class Clip(BaseTransform):
    """Clip a tuple of images to a given range.
    Args:
        mins (scalar or sequence, optional): The minimum values for each channel (default: None).
        maxs (scalar or sequence, optional): The maximum values for each channel (default: None).
    """

    def __init__(self, mins=None, maxs=None):
        super().__init__()
        self._clip = functools.partial(np.clip, a_min=mins, a_max=maxs)

    def __call__(self, *imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be cliped.

        Returns:
            imgs (tuple of numpy.ndarray): The cliped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError('All of the images should be 2D or 3D with channels.')

        imgs = tuple(self._clip(img) for img in imgs)
        return imgs


class Resample(BaseTransform):
    """Resample a tuple of images to the given resolution.
    Args:
        output_spacing (sequence): The target resolution of the images.
    """

    def __init__(self, output_spacing):
        self.output_spacing = output_spacing

    def __call__(self, *imgs, input_spacings, orders=None):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be resampled.
            input_spacings (sequence): The original resolutions of the images.
            orders (sequence of int, optional): The interpolation orders (should be 0, 1 or 3)
                (default: None, the interpolation order would be 1 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The resampled images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError('All of the images should be 2D or 3D with channels.')
        if not all(img.shape[-1] == 1 for img in imgs):
            raise ValueError('All of the images should be single-channel.')

        if orders is None:
            orders = tuple(1 for _ in range(len(imgs)))
        if len(input_spacings) != len(imgs):
            raise ValueError('The number of the input spacings should be the same as the images')
        if len(orders) != len(imgs):
            raise ValueError('The number of the orders should be the same as the images.')
        ndim = imgs[0].ndim
        if not all(len(input_spacing) == (ndim - 1) for input_spacing in input_spacings):
            raise ValueError('The dimensions of all input spacings should be the same as the image.')
        if len(self.output_spacing) != (ndim - 1):
            raise ValueError('The dimensions of the output spacing should be the same as the image.')
        if not all(order in [0, 1, 3] for order in orders):
            raise ValueError('All of the interpolation orders should be 0, 1 or 3.')

        imgs = tuple(self._resample(img, input_spacing, self.output_spacing, order)
                     for img, input_spacing, order in zip(imgs, input_spacings, orders))
        return imgs

    @staticmethod
    def _resample(img, input_spacing, output_spacing, order):
        """Resample the image to the given resolution.
        Args:
            img (np.ndarray): The image to be resampled.
            input_spacing (sequence): The original resolutions of the image.
            output_spacing (sequence): The target resolution of the image.
            order (int): The interpolation order (should be 0, 1 or 3).

        Returns:
            img (np.ndarray): The resampled image.
        """
        resampler = sitk.ResampleImageFilter()
        if order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)

        size = tuple(
            map(
                int,
                (np.array(img.shape[:-1]) * np.array(input_spacing) // np.array(output_spacing))[::-1]
            )
        )
        img = sitk.GetImageFromArray(np.squeeze(img, axis=-1))
        img.SetSpacing(tuple(map(float, input_spacing))[::-1])
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing(tuple(map(float, output_spacing))[::-1])
        resampler.SetSize(size)
        img = resampler.Execute(img)
        img = sitk.GetArrayFromImage(img)[..., np.newaxis]
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
            raise ValueError('All of the images should be 2D or 3D with channels.')
        if not all(img.shape[:-1] == imgs[0].shape[:-1] for img in imgs):
            raise ValueError('All of the images should have the same size.')
        ndim = imgs[0].ndim
        if len(self.size) != (ndim - 1):
            raise ValueError('The dimensions of the cropped size should be the same as the image.')
        if any(i < j for i, j in zip(imgs[0].shape[:-1], self.size)):
            raise ValueError(f'The image size {imgs[0].shape[:-1]} is smaller than '
                             f'the cropped size {self.size}. Please use a smaller cropped size.')

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


class RandomElasticDeform(BaseTransform):
    """Randomly elastic deform a tuple of images by using the bspline transform (as used in U-Net and V-Net).
    Ref:
        https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/rand_elastic_deform.html

    Args:
        do_z_deformation (bool, optional): Whether to apply the deformation along the z dimension (default: False).
            Note that this argument is only valid when images are all 3D.
        num_ctrl_points (int, optional): The number of the control points to form the control point grid (default: 4).
        sigma (scalar, optional): The number to determine the extent of deformation (default: 15).
        prob (float, optional): The probability of applying the deformation (default: 0.5).
    """

    def __init__(self, do_z_deformation=False, num_ctrl_points=4, sigma=15, prob=0.5):
        super().__init__()
        self.do_z_deformation = do_z_deformation
        self.num_ctrl_points = max(num_ctrl_points, 2)
        self.sigma = max(sigma, 1)
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, orders=None):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            orders (sequence of int, optional): The interpolation orders (should be 0, 1 or 3)
                (default: None, the interpolation order would be 3 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The deformed images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError('All of the images should be 2D or 3D with channels.')
        if not all(img.shape[-1] == 1 for img in imgs):
            raise ValueError('All of the images should be single-channel.')
        if not all(img.shape[:-1] == imgs[0].shape[:-1] for img in imgs):
            raise ValueError('All of the images should have the same size.')

        if random.random() < self.prob:
            bspline_transform = self._get_bspline_transform(imgs[0].shape[:-1])
            if orders is None:
                orders = tuple(3 for _ in range(len(imgs)))
            if len(orders) != len(imgs):
                raise ValueError('The number of the orders should be the same as the images.')
            if not all(order in [0, 1, 3] for order in orders):
                raise ValueError('All of the interpolation orders should be 0, 1 or 3.')
            imgs = tuple(self._elastic_deform(img, bspline_transform, order)
                         for img, order in zip(imgs, orders))
        return imgs

    def _get_bspline_transform(self, shape):
        """Get the bspline transform with random parameters.
        Args:
            shape (tuple): The size of the control point grid.
        """
        # Initialize the control point grid.
        img = sitk.GetImageFromArray(np.zeros(shape))
        mesh_size = tuple(self.num_ctrl_points for _ in range(img.GetDimension()))
        bspline_transform = sitk.BSplineTransformInitializer(img, mesh_size)

        # Set the parameters of the bspline transform randomly.
        params = np.array(bspline_transform.GetParameters())
        params = params + np.random.randn(params.shape[0]) * self.sigma
        if len(shape) == 3 and not self.do_z_deformation:
            params[0:len(params) // 3] = 0
        bspline_transform.SetParameters(tuple(params))
        return bspline_transform

    @staticmethod
    def _elastic_deform(img, bspline_transform, order):
        """Elastic deform the image by using the bspline transform.
        Args:
            img (np.ndarray): The image to be deformed.
            bspline_transform (BSplineTransform): The BSplineTransform instance.
            order (int): The interpolation order (should be 0, 1 or 3).

        Returns:
            img (np.ndarray): The deformed image.
        """
        resampler = sitk.ResampleImageFilter()
        if order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)

        img = sitk.GetImageFromArray(np.squeeze(img, axis=-1))
        resampler.SetReferenceImage(img)
        resampler.SetTransform(bspline_transform)
        img = resampler.Execute(img)
        img = sitk.GetArrayFromImage(img)[..., np.newaxis]
        return img


class RandomHorizontalFlip(BaseTransform):
    """Randomly flip a tuple of images horizontally.
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
            raise ValueError('All of the images should be 2D or 3D with channels.')

        if random.random() < self.prob:
            imgs = tuple(np.flip(img, axis=1) for img in imgs)
        return imgs


class RandomVerticalFlip(BaseTransform):
    """Randomly flip a tuple of images vertically.
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
            raise ValueError('All of the images should be 2D or 3D with channels.')

        if random.random() < self.prob:
            imgs = tuple(np.flip(img, axis=0) for img in imgs)
        return imgs
