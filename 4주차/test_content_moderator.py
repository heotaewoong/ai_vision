"""
4주차 Unit Test: 콘텐츠 모더레이션 시스템 검증
pytest 기반
"""

import pytest
import numpy as np
import cv2
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from content_moderator import (
    classify_image_rule_based,
    apply_block_effect,
    LABELS,
    DEFAULT_THRESHOLDS,
)


@pytest.fixture
def safe_image():
    """배경색이 초록(자연)인 안전 이미지"""
    img = np.full((200, 200, 3), [50, 180, 50], dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (160, 160), (30, 120, 30), -1)
    return img


@pytest.fixture
def review_image():
    """살구색 비율이 높은 이미지 (검토 필요 가능성)"""
    img = np.full((200, 200, 3), [160, 140, 200], dtype=np.uint8)
    return img


class TestClassifyRuleBased:
    def test_returns_dict_with_required_keys(self, safe_image):
        result = classify_image_rule_based(safe_image)
        assert "label" in result
        assert "confidence" in result
        assert "all_scores" in result

    def test_label_is_valid_category(self, safe_image):
        result = classify_image_rule_based(safe_image)
        assert result["label"] in LABELS, f"알 수 없는 레이블: {result['label']}"

    def test_confidence_between_0_and_1(self, safe_image):
        result = classify_image_rule_based(safe_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_green_image_is_safe(self, safe_image):
        result = classify_image_rule_based(safe_image)
        assert result["label"] == "SAFE", "초록색 이미지는 SAFE여야 합니다"

    def test_all_black_image_is_safe(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = classify_image_rule_based(black)
        assert result["label"] == "SAFE"

    def test_various_image_sizes(self):
        for h, w in [(50, 50), (224, 224), (480, 640)]:
            img = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
            result = classify_image_rule_based(img)
            assert result["label"] in LABELS


class TestApplyBlockEffect:
    def test_output_same_size_as_input(self, safe_image):
        for verdict in ["SAFE", "UNSAFE", "NEEDS_REVIEW"]:
            out = apply_block_effect(safe_image, verdict)
            assert out.shape == safe_image.shape, f"{verdict}: 크기 불일치"

    def test_output_is_numpy_array(self, safe_image):
        out = apply_block_effect(safe_image, "SAFE")
        assert isinstance(out, np.ndarray)

    def test_unsafe_image_is_heavily_modified(self, safe_image):
        unsafe_out = apply_block_effect(safe_image, "UNSAFE")
        safe_out   = apply_block_effect(safe_image, "SAFE")
        diff = np.mean(np.abs(unsafe_out.astype(float) - safe_image.astype(float)))
        assert diff > 10, "UNSAFE 이미지는 원본과 많이 달라야 합니다 (블러/모자이크)"

    def test_safe_image_preserves_content(self, safe_image):
        safe_out = apply_block_effect(safe_image, "SAFE")
        center_diff = np.mean(np.abs(
            safe_out[50:150, 50:150].astype(float) - safe_image[50:150, 50:150].astype(float)
        ))
        assert center_diff < 50, "SAFE 이미지 중앙부는 크게 변하면 안 됩니다"

    def test_invalid_verdict_handled(self, safe_image):
        # 알 수 없는 레이블도 크래시 없이 처리
        try:
            out = apply_block_effect(safe_image, "UNKNOWN")
            assert out is not None
        except Exception as e:
            pytest.fail(f"알 수 없는 레이블 처리 중 오류: {e}")


class TestLabelsConfig:
    def test_all_required_verdicts_present(self):
        for key in ["SAFE", "UNSAFE", "NEEDS_REVIEW"]:
            assert key in LABELS, f"레이블 누락: {key}"

    def test_each_label_has_color_and_action(self):
        for key, val in LABELS.items():
            assert "color" in val,  f"{key}: color 필드 없음"
            assert "action" in val, f"{key}: action 필드 없음"
            assert len(val["color"]) == 3, f"{key}: color는 (B,G,R) 3채널"

    def test_thresholds_in_valid_range(self):
        for key, val in DEFAULT_THRESHOLDS.items():
            assert 0.0 <= val <= 1.0, f"{key} 임계값이 범위 벗어남: {val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
