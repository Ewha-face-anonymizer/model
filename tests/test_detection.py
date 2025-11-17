import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from components.detection.mtcnn_detector import FaceDetector


def _create_detector():
    return FaceDetector(model_path=None, min_size=20)


def test_load_image_success(tmp_path):
    detector = _create_detector()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), image)

    loaded = detector.load_image(image_path)

    assert loaded.shape == image.shape


def test_load_image_missing(tmp_path):
    detector = _create_detector()
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        detector.load_image(missing)


def test_save_image_creates_parent(tmp_path):
    detector = _create_detector()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    output_path = tmp_path / "nested" / "result.png"

    detector.save_image(image, output_path)

    assert output_path.exists()
    saved = cv2.imread(str(output_path))
    assert saved is not None
