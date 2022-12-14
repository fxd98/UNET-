English | [简体中文](transforms_cn.md)
# [paddleseg.transforms](../../../paddleseg/transforms/transforms.py)

## [Compose](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Compose(transforms, to_rgb = True)
```
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

### Args
* **transforms** (list): A list contains data pre-processing or augmentation.
* **to_rgb** (bool, optional): If converting image to RGB color space. Default: True.

### Raises
* TypeError: When 'transforms' is not a list.
* ValueError: when the length of 'transforms' is less than 1.

## [RandomHorizontalFlip](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomHorizontalFlip(prob = 0.5)
```
    Flip an image horizontally with a certain probability.

### Args
* **prob** (float, optional): A probability of horizontally flipping. Default: 0.5.

## [RandomVerticalFlip](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomVerticalFlip(prob = 0.1)
```
    Flip an image vertically with a certain probability.

### Args
* **prob** (float, optional): A probability of vertical flipping. Default: 0.1.

## [Resize](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Resize(target_size = (512, 512), interp='LINEAR')
```
    Resize an image.

### Args
* **target_size** (list|tuple, optional): The target size of image. Default: (512, 512).
* **interp** (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".

### Raises
* TypeError: When 'target_size' type is neither list nor tuple.
* ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').

## [ResizeByLong](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeByLong(long_size)
```

    Resize the long side of an image to given size, and then scale the other side proportionally.

### Args
* **long_size** (int): The target size of long side.

## [ResizeRangeScaling](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeRangeScaling(min_value = 400, max_value = 600)
```

    Resize the long side of an image into a range, and then scale the other side proportionally.

### Args
* **min_value** (int, optional): The minimum value of long side after resize. Default: 400.
* **max_value** (int, optional): The maximum value of long side after resize. Default: 600.

## [ResizeStepScaling](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeStepScaling(min_scale_factor = 0.75,
                 max_scale_factor = 1.25,
                 scale_step_size = 0.25)
```

    Scale an image proportionally within a range.

### Args
* **min_scale_factor** (float, optional): The minimum scale. Default: 0.75.
* **max_scale_factor** (float, optional): The maximum scale. Default: 1.25.
* **scale_step_size** (float, optional): The scale interval. Default: 0.25.

### Raises
* ValueError: When min_scale_factor is smaller than max_scale_factor.

## [Normalize](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
```

    Normalize an image.The specific operations are as follows:
    1. Min_val is subtracted from the pixel value.
    2. The pixel value is divided by (max_val-min_val) and normalized to the interval [0.0, 1.0].
    3. Subtract the mean value and divide by the standard deviation operation on the image.

    Note: In the mean and std parameters, the length of the list should be consistent with the number of image channels.

### Args
* **mean** (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
* **std** (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

### Raises
* ValueError: When mean/std is not list or any value in std is 0.

## [Padding](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Padding(target_size,
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```
    Perform padding operation on the image or annotated image according to the provided value.Add bottom-right padding to a raw image or annotation image.

### Args
* **target_size** (list|tuple): The target size after padding.
* **im_padding_value** (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
* **label_padding_value** (int, optional): The padding value of annotation image. Default: 255.

### Raises
* TypeError: When target_size is neither list nor tuple.
* ValueError: When the length of target_size is not 2.  

## [RandomPaddingCrop](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomPaddingCrop(crop_size = (512, 512),
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```

    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.

### Args
* **crop_size** (tuple, optional): The target cropping size. Default: (512, 512).
* **im_padding_value** (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
* **label_padding_value** (int, optional): The padding value of annotation image. Default: 255.

### Raises
* TypeError: When crop_size is neither list nor tuple.
* ValueError: When the length of crop_size is not 2.


## [RandomBlur](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomBlur(prob = 0.1)
```

    Blurring an image by a Gaussian function with a certain probability.

### Args
* **prob** (float, optional): A probability of blurring an image. Default: 0.1.

## [RandomRotation](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomRotation(max_rotation = 15,
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```

    Randomly rotate the image. When there is annotated image, it will be synchronized, and the rotated image and the annotated image will be padding accordingly.

### Args
* **max_rotation** (float, optional): The maximum rotation degree. Default: 15.
* **im_padding_value** (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
* **label_padding_value** (int, optional): The padding value of annotation image. Default: 255.
Note: The length of the list of parameters im_padding_value and label_padding_value should be consistent with the number of image channels.

## [RandomScaleAspect](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomScaleAspect(min_scale = 0.5, aspect_ratio = 0.33)
```

    Crop and resize the image back to the original size and the annotated image.
    The image is cropped according to a certain area ratio and aspect ratio, and the reszie returns to the original image. When there is an annotated image, it is synchronized.

### Args
* **min_scale** (float, optional): The minimum value of the area ratio of the cropped image to the original image. The value range is [0, 1]. When it is equal to 0, the original image will be returned.Default: 0.5.
* **aspect_ratio** (float, optional): The minimum value of the cropped image's aspect ratio range, non-negative value, returns to the original image when it is 0.Default: 0.33.


## [RandomDistort](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomDistort(brightness_range = 0.5,
                 brightness_prob = 0.5,
                 contrast_range = 0.5,
                 contrast_prob = 0.5,
                 saturation_range = 0.5,
                 saturation_prob = 0.5,
                 hue_range = 18,
                 hue_prob = 0.5)
```

    Distort an image with random configurations.

### Args
* **brightness_range** (float, optional): A range of brightness. Default: 0.5.
* **brightness_prob** (float, optional): A probability of adjusting brightness. Default: 0.5.
* **contrast_range** (float, optional): A range of contrast. Default: 0.5.
* **contrast_prob** (float, optional): A probability of adjusting contrast. Default: 0.5.
* **saturation_range** (float, optional): A range of saturation. Default: 0.5.
* **saturation_prob** (float, optional): A probability of adjusting saturation. Default: 0.5.
* **hue_range** (int, optional): A range of hue. Default: 18.
* **hue_prob** (float, optional): A probability of adjusting hue. Default: 0.5.
