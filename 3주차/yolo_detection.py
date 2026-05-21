"""
3주차 과제: AI 모델링 및 OpenCV 결과 시각화
YOLOv8 기반 객체 탐지 + Matplotlib 성능 시각화

[비유로 이해하기]
YOLO(You Only Look Once)는 사진을 한 번만 봐도
여러 물체를 동시에 찾아내는 AI입니다.

전통적인 방법(Sliding Window)과 비교하면:
  🐢 전통 방법: 돋보기로 사진을 구석구석 훑어봄 (느림)
  ⚡ YOLO: 사진 전체를 한눈에 파악 (빠름)

의료 AI에서는:
  - 내시경 영상에서 폴립 탐지
  - X-ray에서 이상 부위 박스 표시
  - 현미경 이미지에서 세포 탐지
에 같은 원리를 사용합니다.
"""

import cv2
import numpy as np
import os
import urllib.request
import json
from datetime import datetime

# macOS 한글 폰트 설정 (경고 방지)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def _setup_korean_font():
    """macOS에서 한글 폰트 자동 설정"""
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            matplotlib.rc("font", family=prop.get_name())
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rc("font", family="DejaVu Sans")

_setup_korean_font()


# ─────────────────────────────────────────────
# 1. 모델 로드 (YOLOv8 또는 대체 방법)
# ─────────────────────────────────────────────

def load_yolo_model(model_size: str = "yolov8n"):
    """
    YOLOv8 모델 로드

    모델 크기 옵션 (크기 ↑ = 정확도 ↑ = 속도 ↓):
      yolov8n  - nano  (가장 빠름, 가장 작음)  ← 과제에 적합
      yolov8s  - small
      yolov8m  - medium
      yolov8l  - large
      yolov8x  - extra (가장 정확, 가장 무거움)
    """
    try:
        from ultralytics import YOLO
        model = YOLO(f"{model_size}.pt")
        print(f"  ✅ YOLOv8 모델 로드 성공: {model_size}.pt")
        return model, "yolov8"
    except ImportError:
        print("  ⚠️  ultralytics 미설치 → pip install ultralytics")
        print("  → 대체: OpenCV DNN + COCO 사용")
        return load_opencv_dnn_model()


def load_opencv_dnn_model():
    """
    대체 방법: OpenCV DNN 모듈 + YOLOv3 가중치
    (ultralytics 설치 없이도 동작)
    """
    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

    weights_path = "yolov3-tiny.weights"
    cfg_path = "yolov3-tiny.cfg"
    names_path = "coco.names"

    for url, path in [(weights_url, weights_path), (cfg_url, cfg_path), (names_url, names_path)]:
        if not os.path.exists(path):
            print(f"  다운로드: {os.path.basename(path)} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  ⚠️  다운로드 실패: {e}")
                return None, "failed"

    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path) as f:
        classes = [line.strip() for line in f.readlines()]

    print("  ✅ OpenCV DNN (YOLOv3-tiny) 로드 성공")
    return (net, classes), "opencv_dnn"


# ─────────────────────────────────────────────
# 2. 객체 탐지
# ─────────────────────────────────────────────

def detect_objects_yolov8(model, image: np.ndarray, conf_threshold: float = 0.5) -> list:
    """
    YOLOv8로 객체 탐지

    Returns: list of dict {label, confidence, bbox: (x1,y1,x2,y2)}
    """
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })
    return detections


def detect_objects_opencv_dnn(model_data, image: np.ndarray, conf_threshold: float = 0.5) -> list:
    """
    OpenCV DNN으로 객체 탐지 (대체 방법)
    """
    net, classes = model_data
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    detections = []
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                cx, cy, bw, bh = (detection[:4] * np.array([w, h, w, h])).astype(int)
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    # NMS (Non-Maximum Suppression): 중복 박스 제거
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            detections.append({
                "label": classes[class_ids[i]] if class_ids[i] < len(classes) else "unknown",
                "confidence": confidences[i],
                "bbox": (x, y, x + bw, y + bh),
            })
    return detections


# ─────────────────────────────────────────────
# 3. 결과 시각화 (OpenCV)
# ─────────────────────────────────────────────

# 클래스별 고정 색상 팔레트
COLORS = [
    (0, 255, 0),   (255, 0, 0),   (0, 0, 255),   (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0),  (0, 128, 255),
]

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    탐지 결과를 OpenCV로 시각화

    [비유] 진단 보고서에서 이상 부위에 빨간 박스를 치는 것처럼,
    AI가 찾은 객체에 박스와 레이블을 그립니다.
    """
    vis = image.copy()
    label_color_map = {}

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        # 레이블별 고정 색상
        if label not in label_color_map:
            label_color_map[label] = COLORS[len(label_color_map) % len(COLORS)]
        color = label_color_map[label]

        # 박스 그리기
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 레이블 배경
        text = f"{label}: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # 레이블 텍스트
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 탐지 요약 정보
    summary = f"Detected: {len(detections)} objects | {datetime.now().strftime('%H:%M:%S')}"
    cv2.putText(vis, summary, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def draw_confidence_chart(detections: list, output_path: str):
    """
    탐지된 객체의 신뢰도 막대 차트 저장 (matplotlib)

    [비유] 병원 검사 결과지처럼, 각 항목의 수치를 그래프로 표시합니다.
    """
    if not detections:
        print("  탐지된 객체 없음 → 차트 생략")
        return

    try:
        labels = [f"{d['label']}\n({d['confidence']:.0%})" for d in detections]
        confs = [d["confidence"] for d in detections]
        colors_mpl = ["#2ecc71" if c >= 0.8 else "#f39c12" if c >= 0.6 else "#e74c3c" for c in confs]

        fig, ax = plt.subplots(figsize=(max(8, len(detections) * 1.2), 5))
        bars = ax.bar(labels, confs, color=colors_mpl, edgecolor="black", linewidth=0.5)

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("신뢰도 (Confidence)", fontsize=12)
        ax.set_title("객체 탐지 결과: 클래스별 신뢰도", fontsize=14, fontweight="bold")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="임계값 (0.5)")
        ax.legend()

        for bar, conf in zip(bars, confs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{conf:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  💾 신뢰도 차트 저장: {output_path}")

    except ImportError:
        print("  ⚠️  matplotlib 미설치 → pip install matplotlib")
    except Exception as e:
        print(f"  ⚠️  차트 생성 오류: {e}")


# ─────────────────────────────────────────────
# 4. 모델 성능 평가 시각화
# ─────────────────────────────────────────────

def plot_mock_training_curves(output_path: str):
    """
    모델 학습 곡선 시각화 (시뮬레이션 데이터)

    실제 학습 시에는 model.train() 결과를 사용합니다.
    여기서는 학습 곡선의 형태를 이해하기 위한 예시입니다.

    [비유] 학생이 공부를 많이 할수록 시험 점수가 올라가는 것처럼,
    AI도 학습 횟수(Epoch)가 늘수록 정확도(mAP)가 향상됩니다.
    """
    try:

        epochs = np.arange(1, 21)
        # 시뮬레이션: 로그 형태로 수렴하는 학습 곡선
        train_loss  = 2.0 * np.exp(-0.15 * epochs) + 0.3 + np.random.normal(0, 0.05, 20)
        val_loss    = 2.2 * np.exp(-0.12 * epochs) + 0.4 + np.random.normal(0, 0.08, 20)
        precision   = 1 - 0.8 * np.exp(-0.18 * epochs) + np.random.normal(0, 0.02, 20)
        recall      = 1 - 0.85 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.02, 20)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 곡선
        axes[0].plot(epochs, train_loss, "b-o", markersize=4, label="학습 손실 (Train Loss)")
        axes[0].plot(epochs, val_loss,   "r-s", markersize=4, label="검증 손실 (Val Loss)")
        axes[0].set_title("학습/검증 손실 곡선", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch (학습 횟수)")
        axes[0].set_ylabel("Loss (낮을수록 좋음)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].annotate("수렴 시작", xy=(10, val_loss[9]), fontsize=9,
                         arrowprops=dict(arrowstyle="->"), xytext=(12, 1.0))

        # Precision & Recall
        axes[1].plot(epochs, np.clip(precision, 0, 1), "g-o", markersize=4, label="정밀도 (Precision)")
        axes[1].plot(epochs, np.clip(recall, 0, 1),    "m-s", markersize=4, label="재현율 (Recall)")
        axes[1].set_title("Precision / Recall 곡선", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score (높을수록 좋음)")
        axes[1].set_ylim(0, 1.1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle("YOLOv8 모델 학습 성능 곡선 (시뮬레이션)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  💾 학습 곡선 저장: {output_path}")

    except ImportError:
        print("  ⚠️  matplotlib 미설치")


# ─────────────────────────────────────────────
# 5. 메인 파이프라인
# ─────────────────────────────────────────────

def run_pipeline(image_path: str = None, output_dir: str = "output_detection"):
    print("=" * 60)
    print("  3주차 과제: AI 객체 탐지 파이프라인")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 테스트 이미지 준비
    if image_path is None or not os.path.exists(str(image_path)):
        print("\n[이미지 준비] 샘플 이미지 생성")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 60)
        # 다양한 도형 추가 (객체 시뮬레이션)
        cv2.rectangle(img, (50, 100), (200, 250), (0, 200, 100), -1)
        cv2.rectangle(img, (250, 50), (400, 200), (200, 100, 0), -1)
        cv2.circle(img, (500, 300), 80, (100, 0, 200), -1)
        cv2.ellipse(img, (150, 380), (100, 50), 30, 0, 360, (0, 150, 200), -1)
        image_path = os.path.join(output_dir, "test_input.jpg")
        cv2.imwrite(image_path, img)
        image = img
    else:
        image = cv2.imread(image_path)

    print(f"  이미지: {image_path}  크기: {image.shape}")

    # 모델 로드
    print("\n[1단계] 모델 로드")
    model, model_type = load_yolo_model("yolov8n")

    # 객체 탐지
    print("\n[2단계] 객체 탐지")
    if model_type == "yolov8":
        detections = detect_objects_yolov8(model, image, conf_threshold=0.4)
    elif model_type == "opencv_dnn":
        detections = detect_objects_opencv_dnn(model, image, conf_threshold=0.4)
    else:
        print("  모델 로드 실패 → 더미 탐지 결과 사용")
        detections = [
            {"label": "person", "confidence": 0.92, "bbox": (50, 100, 200, 250)},
            {"label": "car",    "confidence": 0.78, "bbox": (250, 50, 400, 200)},
            {"label": "dog",    "confidence": 0.65, "bbox": (400, 280, 580, 380)},
        ]

    print(f"  탐지된 객체: {len(detections)}개")
    for d in detections:
        print(f"    - {d['label']}: {d['confidence']:.1%}")

    # 결과 저장 요약
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "image_path": str(image_path),
        "total_detections": len(detections),
        "detections": [
            {"label": d["label"], "confidence": round(d["confidence"], 4)}
            for d in detections
        ],
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    # 시각화
    print("\n[3단계] 결과 시각화")
    vis_image = draw_detections(image, detections)
    cv2.imwrite(os.path.join(output_dir, "detection_result.jpg"), vis_image)
    print(f"  💾 탐지 결과 이미지: {output_dir}/detection_result.jpg")

    draw_confidence_chart(detections, os.path.join(output_dir, "confidence_chart.png"))
    plot_mock_training_curves(os.path.join(output_dir, "training_curves.png"))

    print(f"\n✅ 완료! 결과 폴더: {output_dir}/")
    return detections


if __name__ == "__main__":
    run_pipeline()
