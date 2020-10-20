import image_utils as img_u
import os.path
import pickle


# Following staticmethods taken from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/datasets/hmdb51.py

class VideoLoader:
    def __init__(self):
        self.image_loader = img_u.ImageLoader()

    def __call__(self, video_dir_path, frame_indices, file_format):
        return self.video_loader(video_dir_path, frame_indices, file_format)

    def video_loader(self, video_dir_path, frame_indices, file_format):
        """
        Loads a video sequence in a video path, composed by indices. Each frame is loaded using image_loader
        :param video_path:
        :param indices:
        :param image_loader:
        :return:
        """
        video = []
        for frame_index in frame_indices:
            image_path = os.path.join(video_dir_path, f'{frame_index:06d}.{file_format}')
            if os.path.exists(image_path):
                if file_format == "pkl":
                    with open(image_path, "rb") as f:
                        feat_vector = pickle.load(f)
                        img = feat_vector
                else:
                    img = self.image_loader(image_path)
                video.append(img)
            else:
                raise FileNotFoundError(f"image {image_path} does not exist")
        return video