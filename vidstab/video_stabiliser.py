import warnings

import cv2
import numpy as np
from tqdm.auto import trange

from .video import Video


def pts_to_3d_array(pts_list: list) -> np.ndarray:
    """Convert list of 2D points to ndarray of shape (N, 1, 2) of float32.
       This form is used by cv2.findHomography.

    Args:
        pts_list (list): a list of 2D points.

    Returns:
        np.ndarray: the same list as a ndarray of shape (N, 1, 2) of float32.
    """
    return np.array(pts_list, dtype=np.float32).reshape(-1, 1, 2)


class VideoStabiliser:
    """Class for video stabilisation."""

    def __init__(
        self,
        max_matches: int = 100,
        ransac_threshold: float = 0.5,
        focal_length: float = 705.0,
        verbose: bool = False,
    ):
        """Initialise video stabiliser.

        Args:
            max_matches (int, optional): maximum number of matches between
                keypoints to use for aligning frames. Defaults to 100.
            ransac_threshold (float, optional): the threshold argument
                for RANSAC. Defaults to 0.5.
            focal_length (float, optional): focal length. Defaults to 705.
            verbose (bool, optional): sets verbosity level. Defaults to False.
        """
        self.max_matches = max_matches
        self.ransac_threshold = ransac_threshold
        self.focal_length = focal_length
        self.verbose = verbose

        self.quality = []  # calculated after calling stabilise

        self._sift_detector = cv2.SIFT_create()
        self._bf_matcher = cv2.BFMatcher()

    def stabilise(self, video: Video) -> None:
        """Stabilise a video in-place using parameters set in the constructor.

        Args:
            video (Video): video to stabilise.
        """
        self._K = np.array(
            [
                [self.focal_length, 0, video.w / 2],
                [0, self.focal_length, video.h / 2],
                [0, 0, 1],
            ]
        )
        self._K_inv = np.linalg.inv(self._K)

        reference_frame_idx = self._choose_reference_frame(video)
        reference_frame = video[reference_frame_idx]
        kp_ref, des_ref = self._get_keypoints(reference_frame)

        if self.verbose:
            print("Aligning frames...")
        transforms = []
        self.quality = [1e8] * video.frames_count
        counter = range if self.verbose else trange
        for frame_idx in counter(video.frames_count):
            if frame_idx == reference_frame_idx:
                transform = np.eye(3)
            else:
                transform = self._align_frames(
                    (kp_ref, des_ref), video[frame_idx], frame_idx
                )
            transforms.append(transform)

        self.quality[reference_frame_idx] = 0
        if self.verbose:
            mean_quality = np.mean(self.quality)
            print(
                "Stabilisation quality (MSE between keypoints): "
                + f"{mean_quality:.2f}"
            )
        # transforms = self._normalize_trajectory(transforms)  # TODO

        if self.verbose:
            print("Applying transforms...")

        w, h = video.w, video.h
        for i in range(video.frames_count):
            video[i] = cv2.warpPerspective(video[i], transforms[i], (w, h))

        if self.verbose:
            print("Done")

    def _choose_reference_frame(self, video: Video) -> int:
        if self.verbose:
            print("Choosing reference frame...")
        max_sharpness = 0
        max_sharpness_idx = 0
        for i in range(video.frames_count):
            sharpness = self._get_sharpness(video[i])
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                max_sharpness_idx = i
            if self.verbose:
                print(f"Frame {i}: sharpness {sharpness:.1f}")
        if self.verbose:
            print(
                f"Max sharpness: {max_sharpness:.1f}",
                f"at frame {max_sharpness_idx}",
            )
        return max_sharpness_idx

    def _get_sharpness(self, frame: np.ndarray):
        """Returns mean norm of gradient."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(grad_x**2 + grad_y**2).mean()

    def _get_keypoints(self, frame: np.ndarray):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._sift_detector.detectAndCompute(grey, None)

    def _get_matches(self, des_query: np.ndarray, des_train: np.ndarray):
        matches = self._bf_matcher.match(des_query, des_train)
        return sorted(matches, key=lambda x: x.distance)[: self.max_matches]

    def _find_rotation_svd(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> np.ndarray:
        src_mat = src_pts.reshape(-1, 2).T
        dst_mat = dst_pts.reshape(-1, 2).T

        ones = np.ones((1, src_pts.shape[0]))
        src_mat = self._K_inv @ np.vstack([src_mat, ones])
        dst_mat = self._K_inv @ np.vstack([dst_mat, ones.copy()])

        src_mat /= np.linalg.norm(src_mat, axis=0, keepdims=True)
        dst_mat /= np.linalg.norm(dst_mat, axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(src_mat @ dst_mat.T)
        R = vh.T @ u.T
        if np.linalg.det(R) < 0:
            vh[2] *= -1
            R = vh.T @ u.T
        # TODO angles
        return self._K @ R @ self._K_inv

    def _find_camera_rotation_ransac(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        threshold: float = 0.5,
        max_iters: int = 2000,
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_pts = len(src_pts)

        best_rot = np.eye(3)
        best_mask = np.zeros(num_pts)
        most_inliers = 0
        for iter_idx in range(max_iters):
            sample = np.random.choice(len(src_pts), 3, replace=False)
            rotation = self._find_rotation_svd(
                src_pts[sample], dst_pts[sample]
            )

            transformed_pts = cv2.perspectiveTransform(src_pts, rotation)

            error = np.sum((transformed_pts - dst_pts) ** 2, axis=-1)
            mask = error < threshold**2

            inliers = np.sum(mask)
            if inliers > most_inliers:
                best_rot = rotation
                best_mask = mask
                most_inliers = inliers

            if inliers >= confidence * num_pts:
                break

        if self.verbose:
            print("Best inliers:", most_inliers)

        return best_rot, best_mask

    def _align_frames(
        self, reference_frame_data, cur_frame: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        (kp_ref, des_ref) = reference_frame_data
        kp_cur, des_cur = self._get_keypoints(cur_frame)
        matches = self._get_matches(des_ref, des_cur)
        if len(matches) < 4:
            warnings.warn(
                f"Too few matches for frame {frame_idx}. "
                + "Using identity transform"
            )
            return np.eye(3)
        pts_ref = pts_to_3d_array([kp_ref[m.queryIdx].pt for m in matches])
        pts_cur = pts_to_3d_array([kp_cur[m.trainIdx].pt for m in matches])
        H, inlier_mask = self._find_camera_rotation_ransac(
            pts_cur, pts_ref, self.ransac_threshold
        )

        inliers_ref = pts_ref[inlier_mask].reshape(-1, 1, 2)
        inliers_cur = pts_cur[inlier_mask].reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(inliers_cur, H)
        quality = np.mean((projected - inliers_ref).ravel() ** 2)
        self.quality[frame_idx] = quality

        if self.verbose:
            # rx, ry, rz = [float(angle) / np.pi * 180 for angle in angles_rad]
            print(f"Frame {frame_idx}: quality={quality:.2f}")
        return H
