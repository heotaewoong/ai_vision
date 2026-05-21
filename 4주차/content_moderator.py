"""
4주차 과제: 부적절 이미지 자동 분류 및 차단 시스템
Content Moderation AI (콘텐츠 모더레이션)

[실제 사용 기업]
- 카카오, 네이버: 커뮤니티 사진 자동 검열
- YouTube, Instagram: 업로드 전 자동 필터링
- 쿠팡, 무신사: 상품 이미지 품질 검사

[시스템 구성]
  ① 이미지 입력 (파일 or 폴더)
  ② AI 모델 분류 (HuggingFace 사전학습 모델)
  ③ 결과 판정 (SAFE / UNSAFE / NEEDS_REVIEW)
  ④ 차단 처리 + 보고서 출력
  ⑤ OpenCV 시각화 + 로그 저장

[비유]
공항 보안 검색대와 같습니다:
  - X-ray 기기(AI 모델)가 짐(이미지)을 스캔
  - 위험물(부적절 콘텐츠)이 감지되면 경보(UNSAFE) 발생
  - 애매한 경우(NEEDS_REVIEW) → 보안요원(사람)이 최종 판단

[사용 기술]
- HuggingFace Transformers: Falconsai/nsfw_image_detection 모델
- OpenCV: 결과 시각화, 블러 처리 (차단 효과)
- Git + Unit Test: 1·2주차에서 배운 코드 관리 방식 적용
"""

import cv2
import numpy as np
import os
import json
import urllib.request
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# 1. 분류 레이블 및 설정
# ─────────────────────────────────────────────

LABELS = {
    "SAFE":         {"ko": "안전",         "color": (0, 200, 0),   "action": "허용"},
    "UNSAFE":       {"ko": "부적절",        "color": (0, 0, 220),   "action": "차단"},
    "NEEDS_REVIEW": {"ko": "검토 필요",     "color": (0, 165, 255), "action": "보류"},
}

DEFAULT_THRESHOLDS = {
    "safe_min":   0.65,  # 이 이상이면 SAFE
    "unsafe_min": 0.60,  # 이 이상이면 UNSAFE (safe 아닐 때)
    # 나머지는 NEEDS_REVIEW
}


# ─────────────────────────────────────────────
# 2. AI 모델 로드
# ─────────────────────────────────────────────

def load_classifier(model_name: str = "Falconsai/nsfw_image_detection"):
    """
    HuggingFace 사전학습 NSFW 분류 모델 로드

    [비유] 인터넷에서 이미 훈련된 전문가(모델)를 고용하는 것입니다.
    처음부터 수백만 장의 이미지를 직접 학습시킬 필요가 없습니다!
    이것이 Transfer Learning(전이 학습)의 핵심입니다.

    모델 출처: Falconsai/nsfw_image_detection (HuggingFace)
    레이블: normal / nsfw
    """
    try:
        from transformers import pipeline
        from PIL import Image
        classifier = pipeline("image-classification", model=model_name)
        print(f"  ✅ AI 모델 로드 성공: {model_name}")
        return classifier, "huggingface"
    except ImportError:
        print("  ⚠️  transformers / Pillow 미설치")
        print("  → pip install transformers pillow torch torchvision")
        print("  → 대체: 규칙 기반(색상 분석) 분류기 사용")
        return None, "rule_based"
    except Exception as e:
        print(f"  ⚠️  모델 로드 실패: {e}")
        return None, "rule_based"


# ─────────────────────────────────────────────
# 3. 이미지 분류
# ─────────────────────────────────────────────

def classify_image_huggingface(classifier, image: np.ndarray) -> dict:
    """
    HuggingFace 모델로 이미지 분류

    Returns: {"label": str, "confidence": float, "all_scores": dict}
    """
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    results = classifier(pil_img)
    scores = {r["label"]: r["score"] for r in results}

    # "normal" → SAFE, "nsfw" → UNSAFE 매핑
    safe_score   = scores.get("normal", scores.get("safe", 0.0))
    unsafe_score = scores.get("nsfw",   scores.get("unsafe", 0.0))

    if safe_score >= DEFAULT_THRESHOLDS["safe_min"]:
        verdict = "SAFE"
        confidence = safe_score
    elif unsafe_score >= DEFAULT_THRESHOLDS["unsafe_min"]:
        verdict = "UNSAFE"
        confidence = unsafe_score
    else:
        verdict = "NEEDS_REVIEW"
        confidence = max(safe_score, unsafe_score)

    return {"label": verdict, "confidence": confidence, "all_scores": scores}


def classify_image_rule_based(image: np.ndarray) -> dict:
    """
    대체 방법: 색상 분포 분석 기반 규칙 분류기

    [주의] 이 방법은 실제 NSFW 탐지에 적합하지 않습니다.
    단순히 피부색(살구색) 비율을 분석하는 휴리스틱이며,
    False Positive (오탐)가 많습니다.
    실제 서비스에서는 반드시 딥러닝 모델을 사용해야 합니다.

    교육 목적으로만 사용하세요.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 살구색(피부색) HSV 범위 — 조명에 따라 크게 달라짐
    lower_skin = np.array([0,  20, 80],  dtype=np.uint8)
    upper_skin = np.array([25, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])

    # 평균 밝기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0

    if skin_ratio < 0.15:
        verdict, confidence = "SAFE", 0.75
    elif skin_ratio > 0.50 and brightness > 0.4:
        verdict, confidence = "NEEDS_REVIEW", skin_ratio
    else:
        verdict, confidence = "SAFE", 1.0 - skin_ratio

    return {
        "label": verdict,
        "confidence": round(float(confidence), 4),
        "all_scores": {"skin_ratio": round(float(skin_ratio), 4), "brightness": round(float(brightness), 4)},
        "note": "규칙 기반 분류 (참고용, 부정확할 수 있음)",
    }


def classify_image(classifier, model_type: str, image: np.ndarray) -> dict:
    """분류기 통합 인터페이스"""
    if model_type == "huggingface" and classifier is not None:
        return classify_image_huggingface(classifier, image)
    return classify_image_rule_based(image)


# ─────────────────────────────────────────────
# 4. 차단 처리 (OpenCV 시각화)
# ─────────────────────────────────────────────

def apply_block_effect(image: np.ndarray, verdict: str) -> np.ndarray:
    """
    판정 결과에 따른 시각 효과 적용

    SAFE         → 녹색 테두리
    UNSAFE       → 강한 블러 + 빨간 테두리 + 경고 텍스트
    NEEDS_REVIEW → 주황색 테두리 + 반투명 오버레이
    """
    result = image.copy()
    h, w = result.shape[:2]
    info = LABELS.get(verdict, LABELS["NEEDS_REVIEW"])
    color = info["color"]

    if verdict == "UNSAFE":
        # 강한 블러 (모자이크 효과)
        blurred = cv2.GaussianBlur(result, (51, 51), 0)
        # 픽셀화 효과 (모자이크)
        small = cv2.resize(blurred, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        # 빨간 경고 오버레이
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        # 경고 텍스트
        cv2.putText(result, "BLOCKED", (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(result, "Inappropriate Content", (w//2 - 130, h//2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    elif verdict == "NEEDS_REVIEW":
        # 반투명 주황 오버레이
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)

    # 테두리
    border = 8
    cv2.rectangle(result, (border, border), (w - border, h - border), color, border)

    # 판정 배지
    badge_text = f"[{info['ko']}] {info['action']}"
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (8, 8), (bw + 20, bh + 20), color, -1)
    cv2.putText(result, badge_text, (14, bh + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return result


# ─────────────────────────────────────────────
# 5. 배치 처리 + 보고서
# ─────────────────────────────────────────────

def process_folder(
    input_dir: str,
    output_dir: str = "moderation_output",
    classifier=None,
    model_type: str = "rule_based",
):
    """
    폴더 내 이미지 전체 배치 처리 + HTML 보고서 생성

    [비유] 출판사 편집자가 투고된 원고를 전부 검토하고
    '출판 가능', '반려', '수정 필요'로 분류하는 것과 같습니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_dir   = os.path.join(output_dir, "safe")
    unsafe_dir = os.path.join(output_dir, "blocked")
    review_dir = os.path.join(output_dir, "review")
    for d in [safe_dir, unsafe_dir, review_dir]:
        os.makedirs(d, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in exts
    ]

    if not image_files:
        print(f"  ⚠️  {input_dir}에서 이미지를 찾을 수 없습니다.")
        return []

    print(f"\n  총 {len(image_files)}장 처리 시작...")
    logs = []

    for img_path in sorted(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = classify_image(classifier, model_type, image)
        verdict = result["label"]
        processed = apply_block_effect(image, verdict)

        # 저장 경로 결정
        dest_dir = {"SAFE": safe_dir, "UNSAFE": unsafe_dir, "NEEDS_REVIEW": review_dir}[verdict]
        out_path = os.path.join(dest_dir, img_path.name)
        cv2.imwrite(out_path, processed)

        log_entry = {
            "filename":   img_path.name,
            "verdict":    verdict,
            "confidence": result["confidence"],
            "action":     LABELS[verdict]["action"],
            "timestamp":  datetime.now().isoformat(),
        }
        logs.append(log_entry)
        icon = {"SAFE": "✅", "UNSAFE": "🚫", "NEEDS_REVIEW": "⚠️"}[verdict]
        print(f"  {icon} {img_path.name}: {verdict} ({result['confidence']:.0%})")

    # JSON 로그
    log_path = os.path.join(output_dir, "moderation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 요약 통계
    safe_n   = sum(1 for l in logs if l["verdict"] == "SAFE")
    unsafe_n = sum(1 for l in logs if l["verdict"] == "UNSAFE")
    review_n = sum(1 for l in logs if l["verdict"] == "NEEDS_REVIEW")

    print(f"\n{'=' * 50}")
    print(f"  처리 결과 요약")
    print(f"  ✅ 안전:      {safe_n}장")
    print(f"  🚫 차단:      {unsafe_n}장")
    print(f"  ⚠️  검토 필요: {review_n}장")
    print(f"  📋 로그:      {log_path}")
    print(f"{'=' * 50}")

    _generate_summary_chart(logs, os.path.join(output_dir, "summary_chart.png"))
    return logs


def _generate_summary_chart(logs: list, output_path: str):
    """파이 차트 + 막대 그래프로 결과 요약 시각화"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        counts = {"SAFE": 0, "UNSAFE": 0, "NEEDS_REVIEW": 0}
        for l in logs:
            counts[l["verdict"]] = counts.get(l["verdict"], 0) + 1

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 파이 차트
        labels = ["안전 (SAFE)", "차단 (UNSAFE)", "검토 필요"]
        sizes  = [counts["SAFE"], counts["UNSAFE"], counts["NEEDS_REVIEW"]]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]
        explode = (0.05, 0.1, 0.05)
        valid = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
        if valid:
            vs, vl, vc, ve = zip(*valid)
            axes[0].pie(vs, labels=vl, colors=vc, explode=ve, autopct="%1.1f%%",
                        startangle=90, textprops={"fontsize": 11})
        axes[0].set_title("분류 결과 비율", fontsize=13, fontweight="bold")

        # 신뢰도 히스토그램
        confs = [l["confidence"] for l in logs]
        if confs:
            axes[1].hist(confs, bins=10, color="#3498db", edgecolor="black", alpha=0.8)
            axes[1].axvline(x=0.65, color="green", linestyle="--", label="안전 임계값")
            axes[1].axvline(x=0.60, color="red",   linestyle="--", label="차단 임계값")
            axes[1].set_title("신뢰도 분포", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Confidence (신뢰도)")
            axes[1].set_ylabel("이미지 수")
            axes[1].legend()

        plt.suptitle("콘텐츠 모더레이션 결과 요약", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  💾 요약 차트: {output_path}")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# 6. 단일 이미지 빠른 테스트
# ─────────────────────────────────────────────

def quick_test(image_path: str = None):
    """단일 이미지 빠른 분류 테스트"""
    print("=" * 60)
    print("  4주차: 부적절 이미지 분류 시스템 - 테스트 실행")
    print("=" * 60)

    classifier, model_type = load_classifier()

    # 테스트 이미지 생성 (업로드 이미지 없을 때)
    if image_path is None or not os.path.exists(str(image_path)):
        print("\n  테스트 이미지 자동 생성 (실제 서비스에서는 사용자 업로드 이미지 사용)")
        test_images = {
            "safe_test.jpg":   np.full((300, 300, 3), [200, 230, 200], dtype=np.uint8),
            "review_test.jpg": np.full((300, 300, 3), [220, 200, 190], dtype=np.uint8),
        }
        os.makedirs("test_images", exist_ok=True)
        for fname, img_arr in test_images.items():
            cv2.rectangle(img_arr, (50, 50), (250, 250), (100, 150, 100), 3)
            cv2.putText(img_arr, fname.split("_")[0].upper(), (60, 170),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 50, 50), 2)
            cv2.imwrite(f"test_images/{fname}", img_arr)

        image_path = "test_images/safe_test.jpg"

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  이미지 로드 실패: {image_path}")
        return

    print(f"\n  이미지: {image_path}")
    result = classify_image(classifier, model_type, image)
    processed = apply_block_effect(image, result["label"])

    os.makedirs("moderation_output", exist_ok=True)
    cv2.imwrite("moderation_output/quick_test_result.jpg", processed)

    print(f"\n  판정: {result['label']} ({LABELS[result['label']]['ko']})")
    print(f"  신뢰도: {result['confidence']:.1%}")
    print(f"  조치: {LABELS[result['label']]['action']}")
    print(f"  결과: moderation_output/quick_test_result.jpg")


if __name__ == "__main__":
    quick_test()
