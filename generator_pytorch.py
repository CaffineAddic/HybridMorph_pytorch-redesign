import numpy as np
from torch.utils.data import IterableDataset

class IterDataset(IterableDataset):
    def __init__(self, x_data, batch_size=32):
        self.x_data = x_data
        self.batch_size = batch_size

    def __iter__(self):
        """
        Generator that takes in data of size [N, H, W], and yields data for
        our custom vxm model. Note that we need to provide numpy data for each
        input, and each output.

        inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
        outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
        """

        # preliminary sizing
        vol_shape = self.x_data.shape[1:] # extract data shape
        ndims = len(vol_shape)

        # prepare a zero array the size of the deformation
        # we'll explain this below
        zero_phi = np.zeros([self.batch_size, *vol_shape, ndims])

        while True:
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
            idx1 = np.random.randint(0, self.x_data.shape[0], size=self.batch_size)
            moving_images = self.x_data[idx1, ..., np.newaxis]
            idx2 = np.random.randint(0, self.x_data.shape[0], size=self.batch_size)
            fixed_images = self.x_data[idx2, ..., np.newaxis]
            inputs = [moving_images, fixed_images]

            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare
            # the resulting moved image with the fixed image.
            # we also wish to penalize the deformation field.
            outputs = [fixed_images, zero_phi]

            yield (inputs, outputs)