"""
1주차 과제: Git 활용 및 픽셀 단위 이미지 처리 실습
Computer Vision AI 직무 부트캠프

[비유로 이해하기]
이미지 처리는 마치 요리사가 재료를 손질하는 것과 같습니다.
- 이미지 로드    = 냉장고에서 재료 꺼내기
- 크기 조정      = 도마에서 먹기 좋은 크기로 자르기
- 색상 변환      = 재료의 색깔/상태 확인하기
- 노이즈 제거    = 불순물 제거하기
- 데이터 증강    = 같은 재료로 다양한 요리 만들기
"""

import cv2
import numpy as np
import os
import urllib.request

# ─────────────────────────────────────────────
# 0. 샘플 이미지 다운로드 (HuggingFace food101 대신 공개 이미지 사용)
# ─────────────────────────────────────────────
SAMPLE_DIR = "preprocessed_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

def download_sample_images():
    """공개 샘플 이미지 5장 다운로드"""
    urls = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg", "sample_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg", "sample_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg", "sample_03.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png", "sample_04.png"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/320px-Bikesgray.jpg", "sample_05.jpg"),
    ]
    downloaded = []
    for url, fname in urls:
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  ✅ 다운로드: {fname}")
            except Exception as e:
                print(f"  ⚠️  다운로드 실패 ({fname}): {e}")
                # 대체 이미지 생성 (랜덤 색상)
                img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(path, img)
                print(f"  🔄 대체 이미지 생성: {fname}")
        downloaded.append(path)
    return downloaded


# ─────────────────────────────────────────────
# 1. 픽셀 단위 이미지 처리 함수들
# ─────────────────────────────────────────────

def load_and_resize(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    이미지 로드 및 크기 조정 (224×224)

    [비유] 회사 증명사진처럼 규격 크기로 맞추는 작업입니다.
    AI 모델은 항상 같은 크기의 입력을 원하거든요!
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없어요: {image_path}")

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    print(f"  원본 크기: {image.shape[:2]} → 변환 후: {resized.shape[:2]}")
    return resized


def convert_and_normalize(image: np.ndarray) -> tuple:
    """
    색상 변환 + 정규화 (Grayscale & Normalize)

    [비유] 흑백 X-ray처럼 색상 정보를 단순화해서
    AI가 핵심 패턴(모양, 경계선)에 집중하게 합니다.
    정규화는 시험 점수를 0~1 사이로 환산하는 것과 같아요.
    """
    # BGR → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 정규화: 픽셀값 0~255 → 0.0~1.0
    normalized = gray.astype(np.float32) / 255.0

    print(f"  픽셀 범위: 0~255 → 정규화 후: {normalized.min():.2f}~{normalized.max():.2f}")
    return gray, normalized


def remove_noise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    가우시안 블러로 노이즈 제거

    [비유] 사진이 흔들렸을 때 포토샵에서 '부드럽게' 처리하는 것과 같아요.
    의료 영상에서 노이즈를 제거하면 AI가 진짜 병변을 더 잘 찾습니다.
    """
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    print(f"  가우시안 블러 커널 크기: {ksize}×{ksize} 적용 완료")
    return blurred


def detect_color_region(image: np.ndarray, color: str = "red") -> tuple:
    """
    특정 색상 픽셀 감지 및 필터링 (HSV 색공간 활용)

    [비유] 형광펜으로 특정 색만 표시하는 것처럼,
    HSV는 색상(H), 채도(S), 밝기(V)로 색을 구분합니다.
    의료에서는 혈관(빨간색), 종양 표시 등에 활용됩니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [
            (np.array([0, 120, 70]),   np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 120, 70]), np.array([130, 255, 255])),
        ],
        "green": [
            (np.array([40, 80, 40]),   np.array([80, 255, 255])),
        ],
    }

    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in color_ranges.get(color, color_ranges["red"])]
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = cv2.add(combined_mask, m)

    result = cv2.bitwise_and(image, image, mask=combined_mask)
    pixel_count = np.sum(combined_mask > 0)
    print(f"  감지된 {color} 픽셀 수: {pixel_count:,}개")
    return combined_mask, result


def augment_image(image: np.ndarray) -> dict:
    """
    데이터 증강 (Augmentation) - 4가지 변환

    [비유] 같은 X-ray 사진을 뒤집거나 돌려도 여전히 폐 사진이죠?
    AI에게 다양한 각도로 학습시켜서 실전에서 어떤 각도로 찍혀도
    잘 인식하게 만드는 것입니다. (데이터가 부족할 때 특히 유용!)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    augmented = {
        "original":        image.copy(),
        "horizontal_flip": cv2.flip(image, 1),
        "vertical_flip":   cv2.flip(image, 0),
        "rotate_15deg":    cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D(center, 15, 1.0),
            (w, h)
        ),
        "brightness_up":   cv2.convertScaleAbs(image, alpha=1.3, beta=30),
    }
    print(f"  증강 완료: {list(augmented.keys())}")
    return augmented


# ─────────────────────────────────────────────
# 2. 심화: 이상치 탐지 (너무 어두운 이미지 / 작은 객체 제거)
# ─────────────────────────────────────────────

def is_too_dark(image: np.ndarray, threshold: float = 30.0) -> bool:
    """
    평균 밝기 기준으로 너무 어두운 이미지 필터링

    [비유] 병원에서 엑스레이가 너무 흐리게 나오면
    재촬영하는 것처럼, AI도 어두운 이미지는 학습에 방해가 돼요.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = np.mean(gray)
    result = mean_brightness < threshold
    print(f"  평균 밝기: {mean_brightness:.1f} (기준: {threshold}) → {'❌ 제거' if result else '✅ 통과'}")
    return result


def has_valid_object_size(image: np.ndarray, min_ratio: float = 0.01) -> bool:
    """
    엣지 기반으로 유효한 객체 크기 확인

    [비유] 현미경 사진에서 세포가 너무 작으면
    AI가 아무것도 학습하지 못하는 것을 방지합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    result = edge_ratio >= min_ratio
    print(f"  엣지 비율: {edge_ratio:.4f} (최소: {min_ratio}) → {'✅ 유효' if result else '❌ 너무 작음'}")
    return result


# ─────────────────────────────────────────────
# 3. 메인 파이프라인 실행
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("  1주차 과제: 픽셀 단위 이미지 처리 파이프라인")
    print("=" * 60)

    print("\n[1단계] 샘플 이미지 준비 중...")
    image_paths = download_sample_images()

    processed_count = 0

    for idx, path in enumerate(image_paths, 1):
        print(f"\n── 이미지 {idx}: {os.path.basename(path)} ──")

        # 1) 로드 + 리사이즈
        print("[로드 & 크기 조정]")
        try:
            resized = load_and_resize(path, (224, 224))
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        # 2) 이상치 탐지
        print("[이상치 탐지]")
        if is_too_dark(resized):
            print("  → 너무 어두워 건너뜁니다.")
            continue
        if not has_valid_object_size(resized):
            print("  → 객체가 너무 작아 건너뜁니다.")
            continue

        # 3) 색상 변환 + 정규화
        print("[색상 변환 & 정규화]")
        gray, normalized = convert_and_normalize(resized)

        # 4) 노이즈 제거
        print("[노이즈 제거]")
        denoised = remove_noise(resized, ksize=5)

        # 5) 특정 색상 감지
        print("[색상 감지]")
        mask, color_filtered = detect_color_region(resized, "red")

        # 6) 데이터 증강
        print("[데이터 증강]")
        augmented = augment_image(resized)

        # 7) 결과 저장
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(SAMPLE_DIR, f"result_{idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, "resized.jpg"), resized)
        cv2.imwrite(os.path.join(out_dir, "grayscale.jpg"), gray)
        cv2.imwrite(os.path.join(out_dir, "denoised.jpg"), denoised)
        cv2.imwrite(os.path.join(out_dir, "color_mask.jpg"), mask)
        cv2.imwrite(os.path.join(out_dir, "color_filtered.jpg"), color_filtered)
        for aug_name, aug_img in augmented.items():
            cv2.imwrite(os.path.join(out_dir, f"aug_{aug_name}.jpg"), aug_img)

        print(f"  💾 저장 완료: {out_dir}/")
        processed_count += 1

    print(f"\n{'=' * 60}")
    print(f"  처리 완료: {processed_count}/{len(image_paths)} 이미지")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
