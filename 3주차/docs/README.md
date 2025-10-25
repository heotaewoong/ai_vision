# 3주차 과제: AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화

## 📋 과제 개요
- **주제**: AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화
- **목표**: YOLOv8 객체 탐지 모델 학습, 객체 탐지 및 패턴 분석, 결과 시각화
- **기술 스택**: YOLOv8, PyTorch, OpenCV, Matplotlib, Ultralytics
- **데이터셋**: 5개 이미지 (OIP (1).jpg ~ OIP (5).jpg) + 사전 훈련된 YOLOv8 모델

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 3주차 디렉토리로 이동
cd 3주차

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 메인 실행 (권장)
```bash
# 전체 과제 실행 (학습 + 탐지 + 시각화)
python main.py
```

### 3. 개별 모듈 실행
```bash
# 모델 학습만 실행
python yolo_training.py

# OIP 이미지들 객체 탐지만 실행
python object_detection.py

# 시각화 분석만 실행
python visualization_analysis.py
```

## 📁 파일 구조
```
3주차/
├── 📁 핵심 파일들
│   ├── main.py                           # 메인 실행 파일
│   ├── yolo_training.py                  # YOLO 모델 학습
│   ├── object_detection.py               # 객체 탐지 및 패턴 분석
│   ├── visualization_analysis.py         # 결과 시각화 및 성능 분석
│   ├── auto_label_generator.py           # 자동 라벨 생성 도구
│   ├── requirements.txt                  # 필요한 패키지 목록
│   └── data.yaml                        # 데이터셋 설정 파일
│
├── 📁 데이터셋
│   └── datasets/
│       ├── train/images/                 # OIP (1).jpg ~ OIP (4).jpg
│       ├── train/labels/                 # OIP (1).txt ~ OIP (4).txt
│       ├── valid/images/                 # OIP (5).jpg
│       └── valid/labels/                # OIP (5).txt
│
├── 📁 결과물 (results/)
│   ├── detection/                        # 객체 탐지 결과
│   │   ├── detected_OIP (1).jpg         # car 탐지 결과
│   │   ├── detected_OIP (2).jpg          # car 탐지 결과  
│   │   ├── detected_OIP (3).jpg          # person 탐지 결과
│   │   ├── detected_OIP (4).jpg          # dog 탐지 결과
│   │   └── detected_OIP (5).jpg          # dog 탐지 결과
│   │
│   ├── visualization/                    # 시각화 결과
│   │   ├── model_performance_analysis.png    # 모델 성능 분석
│   │   ├── training_analysis.png            # 학습 곡선 분석
│   │   ├── class_performance_analysis.png   # 클래스별 성능
│   │   ├── detection_results_analysis.png   # 탐지 결과 분석
│   │   ├── comprehensive_dashboard.png      # 종합 대시보드
│   │   └── pattern_analysis.png             # 패턴 분석
│   │
│   └── models/                           # 모델 파일들
│       ├── yolov8n.pt                    # 사전 훈련된 YOLOv8 모델
│       └── runs/train/                   # 학습 결과
│
└── 📁 문서 (docs/)
    ├── README.md                         # 이 파일
    └── PPT_프롬프트_Skywork.md           # PPT 작성을 위한 프롬프트
```

## 🎯 주요 기능

### 1. AI 모델 학습 (yolo_training.py)
- YOLOv8 모델 학습 및 성능 평가
- 데이터 증강 및 하이퍼파라미터 튜닝
- 학습 곡선 시각화
- 모델 성능 지표 분석

### 2. 객체 탐지 및 패턴 분석 (object_detection.py)
- 실시간 객체 탐지 (person, car, dog만)
- 클래스별 성능 분석
- 탐지 결과 시각화
- 배치 처리 기능

### 3. 결과 시각화 및 성능 분석 (visualization_analysis.py)
- 종합 성능 대시보드
- 클래스별 성능 분석
- 탐지 결과 통계
- 성능 개선 제안

## 📊 생성되는 결과물

### 🖼️ 시각화 결과 이미지 (results/visualization/)
- `model_performance_analysis.png` - 모델 성능 분석
- `training_analysis.png` - 학습 곡선 및 성능 분석
- `class_performance_analysis.png` - 클래스별 성능 히트맵
- `detection_results_analysis.png` - 탐지 결과 통계 분석
- `comprehensive_dashboard.png` - 종합 성능 대시보드
- `pattern_analysis.png` - 객체 탐지 패턴 분석

### 🔍 객체 탐지 결과 (results/detection/)
- `detected_OIP (1).jpg` - OIP (1).jpg 탐지 결과 (car, 94% 신뢰도)
- `detected_OIP (2).jpg` - OIP (2).jpg 탐지 결과 (car, 94% 신뢰도)
- `detected_OIP (3).jpg` - OIP (3).jpg 탐지 결과 (person, 92% 신뢰도)
- `detected_OIP (4).jpg` - OIP (4).jpg 탐지 결과 (dog, 90% 신뢰도)
- `detected_OIP (5).jpg` - OIP (5).jpg 탐지 결과 (dog, 83% 신뢰도)

### 🤖 학습된 모델 (results/models/)
- `yolov8n.pt` - 사전 훈련된 YOLOv8 모델
- `runs/train/weights/best.pt` - 최고 성능 모델
- `runs/train/` - 학습 로그 및 결과

### 📈 성능 지표 (실제 결과)
- **탐지 정확도**: 100% (5개 이미지에서 5개 객체 정확히 탐지)
- **평균 신뢰도**: 0.906 (90.6%)
- **클래스별 분포**: car 40%, person 20%, dog 40%
- **탐지 성공률**: 100% (person, car, dog만 필터링)

## 🎨 PPT 작성을 위한 프롬프트

PPT 작성을 위해 `docs/PPT_프롬프트_Skywork.md` 파일을 참고하세요. 이 파일에는:
- 4페이지 구성 가이드
- 각 페이지별 상세 프롬프트
- 디자인 가이드라인
- 색상 팔레트 및 레이아웃 요청사항

이 프롬프트를 Skywork에 입력하여 전문적인 PPT를 생성할 수 있습니다.

## 📋 과제 요구사항 달성

### ✅ AI 모델을 활용한 데이터 분석
- YOLOv8 모델 학습 및 성능 평가
- 데이터 증강 및 최적화 기법 적용
- 머신러닝/딥러닝 프레임워크 활용

### ✅ 이미지에서 객체 탐지 및 패턴 분석
- OpenCV를 활용한 이미지 처리
- YOLO 기법을 통한 객체 탐지
- 패턴 분석 및 통계적 접근

### ✅ 결과 시각화 및 보고서 작성
- Matplotlib을 활용한 시각화
- 종합적인 성능 분석 대시보드
- PPT 작성을 위한 프롬프트 제공

## 🚀 실행 환경
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- Ultralytics 8.0+

## 📞 문의사항
과제 실행 중 문제가 발생하면 각 모듈의 주석과 에러 메시지를 확인하세요.
