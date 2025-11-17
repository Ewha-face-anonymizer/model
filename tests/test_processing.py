import pytest

np = pytest.importorskip("numpy")

from components.matching.reference_matcher import ReferenceMatcher
from components.processing.mosaic_processor import MosaicProcessor


def test_reference_matcher_threshold_behaviour():
    matcher = ReferenceMatcher(threshold=0.2)
    reference = np.ones(4, dtype=np.float32)
    positive = reference.copy()
    negative = -reference

    assert matcher.match(positive, reference)
    assert not matcher.match(negative, reference)


def test_reference_matcher_handles_zero_vectors():
    matcher = ReferenceMatcher(threshold=0.3)
    reference = np.zeros(4, dtype=np.float32)
    embedding = np.ones(4, dtype=np.float32)

    assert matcher.match(embedding, reference) is False


def test_mosaic_processor_changes_only_target_roi():
    processor = MosaicProcessor(kernel_size=2)
    frame = np.arange(5 * 5 * 3, dtype=np.uint8).reshape((5, 5, 3))
    bbox = [(0, 0, 3, 3)]

    output = processor.apply(frame, bbox)

    assert not np.array_equal(frame[0:3, 0:3], output[0:3, 0:3])
    assert np.array_equal(frame[3:, :], output[3:, :])
    assert np.array_equal(frame[:, 3:], output[:, 3:])
