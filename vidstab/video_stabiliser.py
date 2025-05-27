import warnings

import cv2
import numpy as np
from tqdm.auto import trange
from scipy.optimize import least_squares

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


def find_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 0.5,
    max_iters: int = 2000,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Find homography matrix using RANSAC.

    Args:
        src_pts (np.ndarray): points in the source image.
        dst_pts (np.ndarray): points in the destination image.
        threshold (float, optional): threshold between inliers and outliers.
            Defaults to 0.5.
        max_iters (int, optional): maximum iterations of random sampling and
            calculating homography. Defaults to 2000.
        confidence (float, optional): the fraction of points to be identified
            as inliers for the algorithm to stop early. Defaults to 0.95.

    Returns:
        tuple[np.ndarray, np.ndarray]: the homography matrix and the mask of
            inliers.
    """
    num_pts = len(src_pts)

    best_H = np.zeros((3, 3))
    best_mask = np.zeros(num_pts)
    most_inliers = 0
    for iter_idx in range(max_iters):
        sample = np.random.choice(len(src_pts), 4, replace=False)
        H = cv2.getPerspectiveTransform(src_pts[sample], dst_pts[sample])

        transformed_pts = cv2.perspectiveTransform(src_pts, H)
        error = np.sum((transformed_pts - dst_pts) ** 2, axis=-1)
        mask = error < threshold**2

        inliers = np.sum(mask)
        if inliers > most_inliers:
            best_H = H
            best_mask = mask
            most_inliers = inliers

        if inliers >= confidence * num_pts:
            break

    return best_H, best_mask.astype(np.uint8)


def camera_rotation_quality(
    angles_rad: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    K: np.ndarray,
    K_inv: np.ndarray,
) -> np.ndarray:
    """Cost function for optimization. Returns an array of residuals."""
    R = rotation_matrix(angles_rad)
    H = K @ R @ K_inv
    projected = cv2.perspectiveTransform(src_pts, H)
    return (projected - dst_pts).ravel()


def rotation_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """Create rotation matrix from Euler angles (Z-Y-X convention)."""
    rx, ry, rz = angles_rad

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


class VideoStabiliser:
    """Class for video stabilisation."""

    def __init__(
        self,
        max_matches: int = 100,
        ransac_threshold: float = 0.5,
        find_angles: bool = True,
        focal_length: float = 705.0,
        verbose: bool = False,
    ):
        """Initialise video stabiliser.

        Args:
            max_matches (int, optional): maximum number of matches between
                keypoints to use for aligning frames. Defaults to 100.
            ransac_threshold (float, optional): the threshold argument
                for RANSAC. Defaults to 0.5.
            find_angles (bool, optional): if True, finds angles and calculates
                the alignment from those. In this case the focal length must be
                provided. After finding the angles, an average rotation is
                calculated and the frames are transformed relative to it.
                If false, finds general homography from 4 points. Average
                rotation is not calculated and focal_length is not used.
                Defaults to True.
            focal_length (float, optional): focal length. Defaults to 705.
            verbose (bool, optional): sets verbosity level. Defaults to False.
        """
        self.max_matches = max_matches
        self.ransac_threshold = ransac_threshold
        self.find_angles = find_angles
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
        if self.find_angles:
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
        counter = range if self.verbose and self.find_angles else trange
        for frame_idx in counter(video.frames_count):
            if frame_idx == reference_frame_idx:
                transform = np.eye(3)
            else:
                transform = self._align_frames(
                    (kp_ref, des_ref), video[frame_idx], frame_idx
                )
            transforms.append(transform)

        if self.find_angles:
            self.quality[reference_frame_idx] = 0
            if self.verbose:
                mean_quality = np.mean(self.quality)
                print(
                    "Stabilisation quality (MSE between keypoints): "
                    + f"{mean_quality:.2f}"
                )
            transforms = self._normalize_trajectory(transforms)

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

    def _angles_to_homography(self, angles_rad: np.ndarray) -> np.ndarray:
        R = rotation_matrix(angles_rad)
        return self._K @ R @ self._K_inv

    def _find_camera_rotation(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        initial_angles = np.zeros(3)
        result = least_squares(
            fun=camera_rotation_quality,
            x0=initial_angles,
            args=(src_pts, dst_pts, self._K, self._K_inv),
        )
        return result.x, result.success

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
        H, mask = find_homography_ransac(
            pts_cur, pts_ref, self.ransac_threshold
        )
        if not self.find_angles:
            return H

        inlier_mask = mask.ravel() == 1
        inliers_ref = pts_ref[inlier_mask]
        inliers_cur = pts_cur[inlier_mask]
        angles_rad, success = self._find_camera_rotation(
            inliers_cur, inliers_ref
        )
        if not success:
            warnings.warn(
                f"Optimization failed for frame {frame_idx}. "
                + "Using transform found by RANSAC"
            )
            return H

        quality = np.mean(
            camera_rotation_quality(
                angles_rad, inliers_cur, inliers_ref, self._K, self._K_inv
            )
            ** 2
        )
        self.quality[frame_idx] = quality

        if self.verbose:
            rx, ry, rz = [float(angle) / np.pi * 180 for angle in angles_rad]
            print(
                f"Frame {frame_idx}: rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}, "
                + f"quality={quality:.2f}"
            )
        return angles_rad

    def _normalize_trajectory(
        self, transforms: list[np.ndarray]
    ) -> list[np.ndarray]:
        normalized_transforms = []
        rotations = []
        for t in transforms:
            if t.ndim == 1:
                normalized_transforms.append(self._angles_to_homography(t))
                rotations.append(t)
            else:
                normalized_transforms.append(t)

        if len(rotations) < len(transforms) // 2:
            return normalized_transforms
        average_rotation = np.mean(np.array(rotations), axis=0)
        if self.verbose:
            rx, ry, rz = [
                float(angle) / np.pi * 180 for angle in average_rotation
            ]
            print(f"Average rotation: rx={rx:.4f}, ry={ry:.4f},  rz={rz:.4f}")
        avg_H = self._angles_to_homography(average_rotation)
        return [np.linalg.inv(avg_H) @ H for H in normalized_transforms]
