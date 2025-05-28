from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image


class TensorboardLogger:
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(str(log_dir))

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # Convert PyTorch tensor to float if necessary
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()  # Convert to plain Python float

        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                if isinstance(img, np.ndarray):
                    # Ensure image has 3 dimensions
                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=-1)
                    if img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB

                    img = img.astype(np.uint8)
                    tf.summary.image(f"{tag}/{i}", np.expand_dims(img, axis=0), step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the values."""
        values = np.array(values)
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
