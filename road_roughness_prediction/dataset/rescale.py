from albumentations import Resize


class Rescale:
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape

        if isinstance(self.output_size, int):
            if min(h, w) > self.output_size:
                return dict(image=image)

            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_w, new_h = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return Resize(new_h, new_w)(image=image)
