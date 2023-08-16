import keras
import numpy as np

class SpinePTXT(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, mask_img_paths,num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mask_img_paths = mask_img_paths
        self.num_classes = num_classes


    def __len__(self):
        return len(self.mask_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_mask_img_paths = self.mask_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size +(3,) , dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            x[j]=img
        y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,), dtype="uint8")
        for j, path in enumerate(batch_mask_img_paths):
            msk = np.load(path)
            y[j]=msk
        return x, y