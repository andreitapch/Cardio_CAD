import os
import cv2
import scipy.io as sio

class VideoToMat:
    def __init__(self, video_path):
        self.video_path = video_path
        # self.cap = cv2.VideoCapture(self.video_path)
        self.cap = cv2.VideoCapture(self.video_path)

    def save_frames_as_mat(self, output_dir):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Create a new directory for each video file
            video_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(self.video_path))[0])
            os.makedirs(video_dir, exist_ok=True)
            mat_file_path = f"{video_dir}/frame_{frame_count}.mat"
            sio.savemat(mat_file_path, {'frame': frame})
            frame_count += 1
        self.cap.release()

root_path = r'D:\PHD\Colaborations\Helena'
output_directory = root_path

for filename in os.listdir(root_path):
    if filename.endswith('.avi'):
        video_path = os.path.join(root_path, filename)
        video_to_mat = VideoToMat(video_path)
        video_to_mat.save_frames_as_mat(output_directory)