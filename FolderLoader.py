import os
import cv2 as cv


# this loads a folder searching for either video or images
class FolderLoader:
    def __init__(self, folder_path):
        self.index = 0
        self.images = []

    def __next__(self):
        if self.index >= len(self.images):
            raise StopIteration

        print(f'type {self}')
        print(f'image: {self.images[self.index]}')

        path = self.images[self.index]
        frame = cv.imread(path)
        self.index += 1
        if frame is None:
            return self.__next__()  # skip unreadable images
        return frame, path

    def __iter__(self):
        return self

    def __repr__(self):
        return "FolderLoader"

class VideoFolderLoader(FolderLoader):
    def __init__(self, folder_path):
        super().__init__(folder_path)
        self.video_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.mp4', '.mov'))])
        self.cap = None

    def __next__(self):
        while True:
            # If no video is currently open, open the next one
            if self.cap is None:
                if self.index >= len(self.video_files):
                    raise StopIteration
                video_path = self.video_files[self.index]
                self.cap = cv.VideoCapture(video_path)
                self.index += 1

            ret, frame = self.cap.read()
            if not ret:
                # Finished this video
                self.cap.release()
                self.cap = None
                continue  # move to next video

            return frame, self.video_files[self.index - 1]

    def __repr__(self):
        return "Video Loader"

class ImageFolderLoader(FolderLoader):
    def __init__(self, folder_path):
        # call parent
        super().__init__(folder_path)
        # do special
        self.images = sorted([os.path.join(folder_path, f)
                              for f in os.listdir(folder_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    def __repr__(self):
        return "Image Loader"