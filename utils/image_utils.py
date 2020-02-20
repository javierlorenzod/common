from PIL import Image
from torchvision import get_image_backend

class ImageLoader:
    def __init__(self):
        if get_image_backend() == 'accimage':
            self.loader_method = ImageLoader.accimage_loader
        else:
            self.loader_method = ImageLoader.pil_loader

    def __call__(self, *args):
        return self.loader_method(*args)

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    @staticmethod
    def accimage_loader(path):
        try:
            import accimage
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return ImageLoader.pil_loader(path)