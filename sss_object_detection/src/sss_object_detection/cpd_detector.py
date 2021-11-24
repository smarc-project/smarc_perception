import numpy as np
import ruptures as rpt
from sss_object_detection.consts import ObjectID


class CPDetector:
    """Change point detector using window sliding for segmentation"""
    def __init__(self):
        self.buoy_width = 19
        self.min_mean_diff_ratio = 1.55

    def detect(self, ping):
        """Detection returns a dictionary with key being ObjectID and
        value being a dictionary of position and confidence of the
        detection."""
        detections = {}

        nadir_idx = self._detect_nadir(ping)
        rope = self._detect_rope(ping, nadir_idx)
        buoy = self._detect_buoy(ping, nadir_idx)

        detections[ObjectID.NADIR] = {'pos': nadir_idx, 'confidence': .9}
        if rope:
            detections[ObjectID.ROPE] = {
                'pos': rope[0][0],
                'confidence': rope[1]
            }
        if buoy:
            detections[ObjectID.BUOY] = {
                'pos': buoy[0][0],
                'confidence': buoy[1]
            }
        return detections

    def _compare_region_with_surrounding(self, ping, bkps, window_size=50):
        region_mean = np.mean(ping[bkps[0]:bkps[1]])
        prev_window = ping[max(bkps[0] - window_size, 0):bkps[0]]
        post_window = ping[bkps[1] + 1:min(bkps[1] +
                                           window_size, ping.shape[0])]
        surrounding_mean = (np.mean(prev_window) + np.mean(post_window)) / 2
        return region_mean / surrounding_mean

    def _detect_rope(self, ping, nadir_idx):
        """Given the tentative nadir_annotation, provide tentative rope
        annotation by segmenting the nadir region. Return None if the
        break point detected is unlikely to be a rope."""
        bkps = self._window_sliding_segmentation(ping=ping,
                                                 start_idx=40,
                                                 end_idx=nadir_idx,
                                                 width=4,
                                                 n_bkps=1)
        bkps = [bkps[0] - 1, bkps[0] + 1]
        mean_diff_ratio = self._compare_region_with_surrounding(ping, bkps)

        if mean_diff_ratio < self.min_mean_diff_ratio:
            return None
        confidence = 1 / mean_diff_ratio
        return bkps, confidence

    def _detect_buoy(self, ping, nadir_idx):
        """Given the tentative nadir_annotation, provide tentative buoy
        detection by segmenting the nadir region. Return None if no
        buoy detected."""
        bkps = self._window_sliding_segmentation(ping=ping,
                                                 start_idx=40,
                                                 end_idx=nadir_idx,
                                                 width=self.buoy_width,
                                                 n_bkps=2)

        # Check whether the segmentation is likely to be a buoy
        if bkps[1] - bkps[0] > self.buoy_width * 2 or bkps[1] - bkps[
                0] < self.buoy_width * .5:
            return None
        mean_diff_ratio = self._compare_region_with_surrounding(ping, bkps)

        if mean_diff_ratio < self.min_mean_diff_ratio:
            return None
        confidence = 1 / mean_diff_ratio
        return bkps, confidence

    def _detect_nadir(self, ping):
        """Use window sliding segmentation to provide tentative
        nadir location annotation. Return detected nadir index."""
        bkps = self._window_sliding_segmentation(ping=ping,
                                                 n_bkps=1,
                                                 start_idx=100,
                                                 end_idx=ping.shape[0],
                                                 width=100)
        return bkps[0]

    def _window_sliding_segmentation(self, ping, n_bkps, start_idx, end_idx,
                                     width):
        """Use window sliding method to segment the input numpy array from
        start_idx to end_idx into (n_bkps + 1) segments. Return a list of
        suggested break points."""

        algo = rpt.Window(width=width, model='l2').fit(ping[start_idx:end_idx])
        bkps = algo.predict(n_bkps=n_bkps)
        bkps = [bkps[i] + start_idx for i in range(len(bkps))]
        return bkps
