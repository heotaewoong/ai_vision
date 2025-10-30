슬라이드 1: 목표 및 개요

- 목표: 로컬 이미지 데이터셋(`dataset/`)으로부터 AI 학습용 전처리 수행
- 요구사항: 224x224로 리사이즈, 그레이스케일 및 정규화, 블러 처리, 데이터 증강(좌우반전, 회전, 색상변형), 이상치 제거(어두운 이미지, 객체가 작은 이미지)

슬라이드 2: 파이프라인 세부사항

- 전처리 순서: 읽기 -> 리사이즈(224x224) -> 블러(Gaussian) -> 그레이스케일 변환 -> 정규화(0-1 또는 0-255 uint8 선택) -> 증강(5종)
- 이상치 판별: 평균 밝기 임계값(기본 35), 최대 컨투어 면적 비율 임계값(기본 0.03)

슬라이드 3: 결과 및 샘플

- 입력: dataset/ 내부의 5개 이미지
- 출력: `submission/samples/` 에 전처리된 5개 샘플 (각각 원본과 증강 중 우선 1개씩 선택)
- 간단 성능: 로컬 CPU에서 5개 이미지 처리(증강 포함) 약 수초 수준

슬라이드 4: 실행 방법 및 참고

- 로컬 실행: `python image_preprocessing.py --input-dir dataset --output-dir preprocessed --save-samples submission/samples`
- 코드 문서화: `submission/README.md` 참조
- (선택) HF 스트리밍 버전은 `image_preprocessing.py`에 있으나, `datasets`/`torch` 의존성으로 인해 별도 설치 필요

슬라이드 5: Git 저장소 구성 및 실습

- 원격 저장소: `https://github.com/heotaewoong/ai_vision.git`
- 로컬 경로: `C:\coding\vision\ai_vision`

초기 설정

```bash
cd C:\coding\vision\ai_vision
git init
git branch -m main
git remote add origin https://github.com/heotaewoong/ai_vision.git
git status
```

초기 커밋 및 푸시

```bash
git add .
git commit -m "chore: initialize project"
git push -u origin main
```

기능 브랜치 생성/작업/푸시

```bash
git checkout -b feature/image-processing
# (코드 수정)
git add -A
git commit -m "feat: add image preprocessing pipeline"
git push -u origin feature/image-processing
```

기존 저장소에 원격만 연결/변경할 때

```bash
git remote -v
git remote set-url origin https://github.com/heotaewoong/ai_vision.git
```

권장 .gitignore (요약)

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/

# Data & results
1주차/dataset/
3주차/results/

# OS/editor
.DS_Store
*.code-workspace
.idea/
.vscode/
```

(선택) 큰 이미지/모델 파일은 Git LFS 사용

```bash
git lfs install
git lfs track "*.jpg" "*.png" "*.pt"
git add .gitattributes
```
