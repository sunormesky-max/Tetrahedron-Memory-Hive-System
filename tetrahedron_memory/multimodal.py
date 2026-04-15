"""
Multimodal mapping module for Tetrahedron Memory System.
"""

import hashlib
from typing import List

import numpy as np


class PixHomology:
    def __init__(self, resolution: int = 32):
        self.resolution = resolution

    def image_to_geometry(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        if image.shape[0] != self.resolution or image.shape[1] != self.resolution:
            from scipy.ndimage import zoom

            zoom_factor = (self.resolution / image.shape[0], self.resolution / image.shape[1])
            image = np.array(zoom(image, zoom_factor))

        features = self._extract_topological_features(image)

        if len(features) >= 3:
            point = np.array(features[:3], dtype=np.float64)
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm
            return point
        else:
            hash_val = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
            rng = np.random.RandomState(hash_val % (2**31))

            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * rng.uniform(0, 1) - 1)

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            return np.array([x, y, z], dtype=np.float64)

    def _extract_topological_features(self, image: np.ndarray) -> List[float]:
        try:
            import gudhi

            points = []
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if image[i, j] > 0.5:
                        points.append([i, j, image[i, j]])

            if len(points) < 4:
                return []

            points = np.array(points)

            alpha_complex = gudhi.AlphaComplex(points=points.tolist())
            simplex_tree = alpha_complex.create_simplex_tree()

            simplex_tree.persistence()

            features = []
            for interval in simplex_tree.persistence_intervals_in_dimension(0):
                birth, death = interval
                length = death - birth
                features.append(length)

            return features[:10]
        except ImportError:
            return []

    def image_to_tetrahedron(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        if image.shape[0] != self.resolution or image.shape[1] != self.resolution:
            from scipy.ndimage import zoom

            zoom_factor = (self.resolution / image.shape[0], self.resolution / image.shape[1])
            image = np.array(zoom(image, zoom_factor))

        features = self._extract_topological_features(image)

        vertices = []
        if len(features) >= 12:
            for i in range(4):
                vertex = np.array(features[i * 3 : (i + 1) * 3])
                if len(vertex) == 3:
                    norm = np.linalg.norm(vertex)
                    if norm > 0:
                        vertex = vertex / norm
                    vertices.append(vertex)

        if len(vertices) < 4:
            for i in range(4):
                seed_data = image.tobytes() + str(i).encode()
                hash_val = int(hashlib.md5(seed_data).hexdigest()[:8], 16)
                rng = np.random.RandomState(hash_val % (2**31))

                theta = rng.uniform(0, 2 * np.pi)
                phi = np.arccos(2 * rng.uniform(0, 1) - 1)

                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)

                vertices.append([x, y, z])

        return np.array(vertices, dtype=np.float64)

    def audio_to_geometry(self, audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)

        n_samples = len(audio_data)
        frame_size = min(512, n_samples)
        hop = frame_size // 2
        n_frames = max(1, (n_samples - frame_size) // hop + 1)

        window = np.hanning(frame_size)
        mfcc_features = []
        for i in range(n_frames):
            start = i * hop
            frame = audio_data[start : start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frame = frame * window
            spectrum = np.abs(np.fft.rfft(frame))
            mel_weights = self._mel_filterbank(frame_size, sample_rate, n_filters=13)
            mel_spec = mel_weights @ spectrum[: mel_weights.shape[1]]
            mel_spec = np.log(mel_spec + 1e-12)
            dct_coeffs = np.zeros(13)
            for k in range(13):
                dct_coeffs[k] = np.sum(mel_spec * np.cos(np.pi * k * (np.arange(13) + 0.5) / 13))
            mfcc_features.append(dct_coeffs[:3])

        if not mfcc_features:
            return self._fallback_geometry(audio_data.tobytes())

        point_cloud = np.array(mfcc_features)
        features = self._run_ph_on_points(point_cloud)

        if len(features) >= 3:
            point = np.array(features[:3], dtype=np.float64)
            norm = np.linalg.norm(point)
            return point / norm if norm > 0 else point
        return self._fallback_geometry(audio_data.tobytes())

    def _mel_filterbank(self, fft_size: int, sample_rate: int, n_filters: int = 13) -> np.ndarray:
        low_mel = 0.0
        high_mel = 1127 * np.log1p((sample_rate / 2) / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = 700 * (np.exp(mel_points / 1127) - 1)
        bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
        n_fft_bins = fft_size // 2 + 1
        filterbank = np.zeros((n_filters, n_fft_bins))
        for i in range(n_filters):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            for j in range(left, center):
                if j < n_fft_bins and center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if j < n_fft_bins and right > center:
                    filterbank[i, j] = (right - j) / (right - center)
        return filterbank

    def video_to_geometry(self, frames: List[np.ndarray], fps: float = 30.0) -> np.ndarray:
        if not frames:
            return np.array([0.0, 0.0, 1.0])
        step = max(1, len(frames) // 8)
        key_frames = frames[::step][:8]
        geometries = [self.image_to_geometry(f) for f in key_frames]
        if not geometries:
            return np.array([0.0, 0.0, 1.0])
        avg = np.mean(geometries, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def _run_ph_on_points(self, points: np.ndarray) -> List[float]:
        try:
            import gudhi

            if len(points) < 4:
                return []
            alpha = gudhi.AlphaComplex(points=points.tolist())
            st = alpha.create_simplex_tree()
            st.persistence()
            return [iv[1] - iv[0] for iv in st.persistence_intervals_in_dimension(0)]
        except Exception:
            return []

    def _fallback_geometry(self, data: bytes) -> np.ndarray:
        hash_val = int(hashlib.md5(data).hexdigest()[:8], 16)
        rng = np.random.RandomState(hash_val % (2**31))
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * rng.uniform(0, 1) - 1)
        return np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ],
            dtype=np.float64,
        )


from .geometry import TextToGeometryMapper  # noqa: E402, F401 — re-export for backward compat
