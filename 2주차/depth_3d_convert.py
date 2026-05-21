"""
2주차 과제: Unit Test 구성 및 2D → 3D 변환 알고리즘 (개선판)
Computer Vision AI 직무 부트캠프

[이전 코드의 문제점]
기존 코드는 그레이스케일 + JET 컬러맵을 "Depth Map"이라고 불렀는데,
이건 진짜 깊이 추정이 아닙니다! 단순히 밝기를 색으로 표현한 것뿐입니다.

[진짜 Depth Estimation이란?]
카메라에서 각 픽셀까지의 실제 거리(깊이)를 추정하는 기술입니다.

[비유로 이해하기]
- 2D 이미지 = 지도 (위에서 내려다본 평면 그림)
- Depth Map  = 지도 위의 등고선 (높낮이 정보)
- 3D Point Cloud = 실제 지형 모형 (입체적인 조각상)

의료 AI에서는 CT/MRI의 2D 슬라이스들을 합쳐서 3D 장기 모델을 만들 때 씁니다!

[이번 과제에서 사용하는 방법]
MiDaS (by Intel): 단일 이미지에서 깊이를 추정하는 딥러닝 모델
PyTorch Hub를 통해 간단하게 사용 가능합니다.
(인터넷 없을 경우 → 대체 방법: 스테레오 비전 또는 라플라시안 기반 추정 사용)
"""

import cv2
import numpy as np
import os
import urllib.request


# ─────────────────────────────────────────────
# 1. Depth Map 생성 (MiDaS 또는 대체 방법)
# ─────────────────────────────────────────────

def generate_depth_map_midas(image: np.ndarray) -> np.ndarray:
    """
    MiDaS 딥러닝 모델로 진짜 Depth Map 생성

    [비유] 사람이 두 눈으로 입체감을 느끼듯,
    MiDaS는 하나의 카메라로 깊이를 추정합니다.
    빛의 반사, 그림자, 원근법을 딥러닝으로 학습했어요.
    """
    try:
        import torch

        # MiDaS 모델 로드 (작은 버전 사용 - 속도 빠름)
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.eval()

        # 변환기 로드
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        # 이미지 전처리
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb)

        # 깊이 추정
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.numpy()
        # 0~255로 정규화
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print("  ✅ MiDaS 딥러닝 깊이 추정 성공!")
        return depth_norm

    except Exception as e:
        print(f"  ⚠️  MiDaS 사용 불가 ({type(e).__name__}): {e}")
        print("  → 대체 방법: 라플라시안 + 엣지 기반 깊이 추정 사용")
        return generate_depth_map_fallback(image)


def generate_depth_map_fallback(image: np.ndarray) -> np.ndarray:
    """
    대체 방법: 라플라시안 + 거리 변환 기반 깊이 추정

    [비유] 사진에서 선명한 부분(엣지)은 가까이 있고,
    흐릿한 부분은 멀리 있다는 원리를 활용합니다.
    보케(아웃포커스) 효과와 반대 원리입니다!

    이 방법은 진짜 MiDaS보다 정확도가 낮지만,
    인터넷 없이도 즉시 사용 가능합니다.
    """
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 라플라시안으로 엣지(선명도) 검출 - 가까운 물체일수록 선명함
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.abs(laplacian)

    # 가우시안 블러로 부드럽게 (거리감 표현)
    blurred = cv2.GaussianBlur(sharpness, (21, 21), 0)
    depth_map = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return depth_map


def visualize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Depth Map을 JET 컬러맵으로 시각화

    [비유] 날씨 지도에서 온도를 파랑→빨강으로 표현하듯,
    깊이도 파랑(멀리) → 빨강(가까이)으로 표현합니다.
    """
    colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return colored


# ─────────────────────────────────────────────
# 2. 3D 포인트 클라우드 생성
# ─────────────────────────────────────────────

def create_3d_point_cloud(
    image: np.ndarray,
    depth_map: np.ndarray,
    focal_length: float = 500.0,
    max_points: int = 50000,
) -> np.ndarray:
    """
    Depth Map + 원본 이미지로 3D 포인트 클라우드 생성

    [비유] 구글 스트리트뷰 3D 뷰어처럼,
    각 픽셀의 위치(X,Y)와 깊이(Z)를 합쳐 3D 공간상의 점으로 만듭니다.
    결과물은 (X, Y, Z, R, G, B) 좌표+색상 정보입니다.

    의료에서는 CT 슬라이스에서 이런 방식으로 3D 장기를 재구성합니다!

    Parameters
    ----------
    focal_length : 카메라 초점 거리 (픽셀 단위, 기본값 500)
    max_points   : 최대 포인트 수 (메모리 절약)
    """
    if image is None or depth_map is None:
        raise ValueError("이미지 또는 Depth Map이 없습니다.")

    h, w = depth_map.shape[:2]
    cx, cy = w / 2.0, h / 2.0  # 이미지 중심점 (주점)

    # 픽셀 격자 생성
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Z축: 깊이값 (0 제외)
    Z = depth_map.astype(np.float32)
    valid = Z > 0

    # 핀홀 카메라 모델: X = (u - cx) * Z / f
    X = (u[valid] - cx) * Z[valid] / focal_length
    Y = (v[valid] - cy) * Z[valid] / focal_length
    Zv = Z[valid]

    # 색상 추출 (BGR → RGB)
    if len(image.shape) == 3:
        B = image[:, :, 0][valid].astype(np.float32)
        G = image[:, :, 1][valid].astype(np.float32)
        R = image[:, :, 2][valid].astype(np.float32)
        colors = np.stack([R, G, B], axis=1) / 255.0
    else:
        gray_vals = image[valid].astype(np.float32) / 255.0
        colors = np.stack([gray_vals, gray_vals, gray_vals], axis=1)

    points = np.stack([X, Y, Zv], axis=1)
    point_cloud = np.concatenate([points, colors], axis=1)  # [X, Y, Z, R, G, B]

    # 포인트 수 제한 (랜덤 샘플링)
    if len(point_cloud) > max_points:
        idx = np.random.choice(len(point_cloud), max_points, replace=False)
        point_cloud = point_cloud[idx]

    print(f"  생성된 3D 포인트: {len(point_cloud):,}개")
    return point_cloud


def save_point_cloud_ply(point_cloud: np.ndarray, output_path: str):
    """
    3D 포인트 클라우드를 PLY 파일로 저장 (CloudCompare, MeshLab에서 열 수 있음)

    PLY 포맷은 3D 점의 위치와 색상을 저장하는 표준 형식입니다.
    """
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(point_cloud)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(output_path, "w") as f:
        f.write(header)
        for pt in point_cloud:
            x, y, z, r, g, b = pt
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r*255)} {int(g*255)} {int(b*255)}\n")
    print(f"  💾 PLY 저장 완료: {output_path}")


def save_point_cloud_txt(point_cloud: np.ndarray, output_path: str):
    """XYZ RGB 형식의 텍스트 파일로 저장"""
    np.savetxt(
        output_path,
        point_cloud,
        fmt="%.4f %.4f %.4f %.4f %.4f %.4f",
        header="X Y Z R G B",
        comments="",
    )
    print(f"  💾 TXT 저장 완료: {output_path}")


# ─────────────────────────────────────────────
# 3. 메인 실행
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = "output_3d"):
    """2D→3D 변환 전체 파이프라인"""
    print("=" * 60)
    print("  2주차 과제: 2D → 3D 변환 파이프라인")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    print("\n[1단계] 이미지 로드")
    image = cv2.imread(image_path)
    if image is None:
        # 테스트용 이미지 자동 생성
        print("  이미지 없음 → 테스트 이미지 자동 생성")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (0, 100, 200), -1)
        cv2.circle(image, (450, 200), 80, (200, 50, 50), -1)
        cv2.imwrite(image_path, image)
    print(f"  이미지 크기: {image.shape}")

    # Depth Map 생성
    print("\n[2단계] Depth Map 생성 (MiDaS or 대체)")
    depth_map = generate_depth_map_midas(image)
    depth_colored = visualize_depth_map(depth_map)

    # 3D 포인트 클라우드 생성
    print("\n[3단계] 3D 포인트 클라우드 생성")
    point_cloud = create_3d_point_cloud(image, depth_map, max_points=30000)

    # 결과 저장
    print("\n[4단계] 결과 저장")
    cv2.imwrite(os.path.join(output_dir, "depth_map_colored.jpg"), depth_colored)
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), image)
    save_point_cloud_ply(point_cloud, os.path.join(output_dir, "point_cloud.ply"))
    save_point_cloud_txt(point_cloud, os.path.join(output_dir, "point_cloud.txt"))

    print(f"\n✅ 완료! 결과 폴더: {output_dir}/")
    return depth_map, point_cloud


if __name__ == "__main__":
    run_pipeline("sample.jpg")
