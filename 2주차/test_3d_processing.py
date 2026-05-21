"""
2주차 과제: Unit Test - 2D→3D 변환 코드 검증
pytest 기반

[Unit Test가 왜 필요한가?]
비유: 의사가 처방전을 쓰기 전에 약의 성분을 검사하는 것처럼,
코드도 실제 사용 전에 각 기능이 정상 작동하는지 검증해야 합니다.

Unit Test = 코드의 '부품 검사'
- 부품(함수) 하나씩 테스트
- 이상한 입력이 들어와도 안전하게 처리되는지 확인
- 코드를 수정해도 기존 기능이 망가지지 않았는지 확인 (회귀 테스트)
"""

import pytest
import numpy as np
import cv2
import os
import sys

# 같은 폴더의 depth_3d_convert 모듈 import
sys.path.insert(0, os.path.dirname(__file__))
from depth_3d_convert import (
    generate_depth_map_fallback,
    visualize_depth_map,
    create_3d_point_cloud,
    save_point_cloud_txt,
)


# ─────────────────────────────────────────────
# 공통 픽스처 (테스트에서 공유할 샘플 데이터)
# ─────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """100×100 픽셀 테스트 이미지 (컬러, 다양한 영역 포함)"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:50, 20:50] = [200, 100, 50]   # 왼쪽 위: 색상 블록
    img[60:90, 60:90] = [50, 200, 100]   # 오른쪽 아래: 다른 색상
    return img


@pytest.fixture
def sample_depth(sample_image):
    """샘플 이미지로 생성한 Depth Map"""
    return generate_depth_map_fallback(sample_image)


# ─────────────────────────────────────────────
# 테스트 1: generate_depth_map_fallback
# ─────────────────────────────────────────────

class TestGenerateDepthMap:
    """Depth Map 생성 함수 테스트"""

    def test_output_shape_matches_input(self, sample_image):
        """출력 크기가 입력 이미지 크기와 동일해야 함"""
        depth = generate_depth_map_fallback(sample_image)
        assert depth.shape[:2] == sample_image.shape[:2], (
            f"크기 불일치: 입력={sample_image.shape[:2]}, 출력={depth.shape[:2]}"
        )

    def test_output_dtype_is_uint8(self, sample_image):
        """출력 타입이 uint8 이어야 함 (0~255 범위)"""
        depth = generate_depth_map_fallback(sample_image)
        assert depth.dtype == np.uint8, f"타입 불일치: {depth.dtype} (기대: uint8)"

    def test_pixel_range_0_to_255(self, sample_image):
        """픽셀 값이 0~255 범위 안에 있어야 함"""
        depth = generate_depth_map_fallback(sample_image)
        assert depth.min() >= 0, f"최솟값 오류: {depth.min()}"
        assert depth.max() <= 255, f"최댓값 오류: {depth.max()}"

    def test_output_is_2d_grayscale(self, sample_image):
        """Depth Map은 2D 흑백 이미지여야 함"""
        depth = generate_depth_map_fallback(sample_image)
        assert len(depth.shape) == 2, f"차원 오류: {len(depth.shape)}D (기대: 2D)"

    def test_none_input_raises_value_error(self):
        """None 입력 시 ValueError 발생해야 함"""
        with pytest.raises(ValueError, match="입력 이미지가 없습니다"):
            generate_depth_map_fallback(None)

    def test_various_sizes(self):
        """다양한 크기의 이미지 처리 가능해야 함"""
        for h, w in [(50, 50), (100, 200), (224, 224), (480, 640)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            depth = generate_depth_map_fallback(img)
            assert depth.shape == (h, w), f"크기 처리 실패: {h}×{w}"

    def test_all_black_image(self):
        """완전 검정 이미지도 오류 없이 처리해야 함"""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = generate_depth_map_fallback(black)
        assert depth is not None
        assert depth.shape == (100, 100)

    def test_all_white_image(self):
        """완전 흰색 이미지도 오류 없이 처리해야 함"""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        depth = generate_depth_map_fallback(white)
        assert depth is not None


# ─────────────────────────────────────────────
# 테스트 2: visualize_depth_map
# ─────────────────────────────────────────────

class TestVisualizeDepthMap:
    """Depth Map 시각화 함수 테스트"""

    def test_output_is_color_image(self, sample_depth):
        """출력이 3채널 컬러 이미지여야 함 (JET 컬러맵)"""
        colored = visualize_depth_map(sample_depth)
        assert len(colored.shape) == 3, "출력이 3D 배열이어야 함"
        assert colored.shape[2] == 3, "3채널(BGR)이어야 함"

    def test_output_shape_matches_input(self, sample_depth):
        """공간 크기(H, W)가 입력과 동일해야 함"""
        colored = visualize_depth_map(sample_depth)
        assert colored.shape[:2] == sample_depth.shape[:2]

    def test_not_all_zeros(self, sample_image):
        """시각화 결과가 완전 검정(0)이 아니어야 함"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        depth = generate_depth_map_fallback(img)
        colored = visualize_depth_map(depth)
        assert colored.max() > 0, "시각화 결과가 비어있음"


# ─────────────────────────────────────────────
# 테스트 3: create_3d_point_cloud
# ─────────────────────────────────────────────

class TestCreate3DPointCloud:
    """3D 포인트 클라우드 생성 함수 테스트"""

    def test_output_shape_has_6_columns(self, sample_image, sample_depth):
        """각 포인트가 [X, Y, Z, R, G, B] 6개 값을 가져야 함"""
        pc = create_3d_point_cloud(sample_image, sample_depth)
        assert pc.shape[1] == 6, f"열 수 오류: {pc.shape[1]} (기대: 6)"

    def test_point_count_is_positive(self, sample_image, sample_depth):
        """최소 1개 이상의 포인트가 생성되어야 함"""
        pc = create_3d_point_cloud(sample_image, sample_depth)
        assert len(pc) > 0, "포인트 클라우드가 비어있음"

    def test_max_points_limit_respected(self, sample_image, sample_depth):
        """max_points 제한이 지켜져야 함"""
        max_pts = 500
        pc = create_3d_point_cloud(sample_image, sample_depth, max_points=max_pts)
        assert len(pc) <= max_pts, f"포인트 수 초과: {len(pc)} > {max_pts}"

    def test_color_values_in_range_0_1(self, sample_image, sample_depth):
        """색상 값(R, G, B)이 0.0~1.0 범위여야 함"""
        pc = create_3d_point_cloud(sample_image, sample_depth)
        colors = pc[:, 3:]
        assert colors.min() >= 0.0, f"색상 최솟값 오류: {colors.min()}"
        assert colors.max() <= 1.0, f"색상 최댓값 오류: {colors.max()}"

    def test_none_image_raises_error(self, sample_depth):
        """image=None 입력 시 ValueError 발생해야 함"""
        with pytest.raises(ValueError):
            create_3d_point_cloud(None, sample_depth)

    def test_none_depth_raises_error(self, sample_image):
        """depth_map=None 입력 시 ValueError 발생해야 함"""
        with pytest.raises(ValueError):
            create_3d_point_cloud(sample_image, None)

    def test_grayscale_image_also_works(self, sample_depth):
        """흑백 이미지도 처리 가능해야 함"""
        gray_img = np.full((100, 100), 128, dtype=np.uint8)
        pc = create_3d_point_cloud(gray_img, sample_depth)
        assert pc.shape[1] == 6


# ─────────────────────────────────────────────
# 테스트 4: save_point_cloud_txt
# ─────────────────────────────────────────────

class TestSavePointCloud:
    """포인트 클라우드 저장 함수 테스트"""

    def test_file_is_created(self, sample_image, sample_depth, tmp_path):
        """파일이 실제로 생성되어야 함"""
        pc = create_3d_point_cloud(sample_image, sample_depth, max_points=100)
        out_path = str(tmp_path / "test_pc.txt")
        save_point_cloud_txt(pc, out_path)
        assert os.path.exists(out_path), "파일이 생성되지 않음"

    def test_file_is_not_empty(self, sample_image, sample_depth, tmp_path):
        """생성된 파일이 비어있지 않아야 함"""
        pc = create_3d_point_cloud(sample_image, sample_depth, max_points=100)
        out_path = str(tmp_path / "test_pc.txt")
        save_point_cloud_txt(pc, out_path)
        assert os.path.getsize(out_path) > 0, "파일이 비어있음"

    def test_line_count_matches_point_count(self, sample_image, sample_depth, tmp_path):
        """파일의 데이터 행 수가 포인트 수와 일치해야 함 (헤더 1행 제외)"""
        pc = create_3d_point_cloud(sample_image, sample_depth, max_points=50)
        out_path = str(tmp_path / "test_pc.txt")
        save_point_cloud_txt(pc, out_path)
        with open(out_path) as f:
            lines = [l for l in f.readlines() if l.strip() and not l.startswith("X")]
        assert len(lines) == len(pc), f"행 수 불일치: {len(lines)} != {len(pc)}"


# ─────────────────────────────────────────────
# 실행 진입점
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("pytest로 Unit Test 실행 중...")
    pytest.main([__file__, "-v", "--tb=short"])
