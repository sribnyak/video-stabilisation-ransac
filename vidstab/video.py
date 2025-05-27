import cv2
import numpy as np


class Video:
    """Class for videos without sound."""

    def __init__(self, frames: np.ndarray, fps: float):
        """Create a video object from frames and fps.
        Videos can be also created by opening video files with open_video.

        Args:
            frames (np.ndarray): array of video's frames. Must be not empty.
            fps (float): FPS.
        """
        assert len(frames) > 0, "Video is empty!"
        self._frames = frames
        self.fps = fps

    @property
    def frames_count(self) -> int:
        """Number of frames in the video."""
        return self._frames.shape[0]

    @property
    def h(self) -> int:
        """Height of the video."""
        return self._frames.shape[1]

    @property
    def w(self) -> int:
        """Width of the video."""
        return self._frames.shape[2]

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get frame by index."""
        return self._frames[idx]

    def __setitem__(self, idx: int, val: np.ndarray):
        """Set frame by index."""
        self._frames[idx] = val

    def save(self, path: str, fourcc_code: str = "mp4v") -> None:
        """Save video to file.

        Args:
            path (str): path to save video to.
            fourcc_code (str, optional): see cv2.VideoWriter_fourcc.
                Defaults to "mp4v".
        """
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)  # type: ignore
        out = cv2.VideoWriter(path, fourcc, self.fps, (self.w, self.h))

        for frame in self._frames:
            out.write(frame)
        out.release()


def open_video(video_path: str) -> Video:
    """Open video file.
    Note: the sound is not supported. If the video has sound, it will be lost.

    Args:
        video_path (str): path to video file.

    Returns:
        Video: the opened video (without sound).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return Video(np.array(frames), fps)
