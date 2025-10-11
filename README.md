# 이미지 전처리 (Task 1 추가 과제)

이 저장소에는 Hugging Face 데이터셋 또는 로컬 이미지 폴더에서 이미지를 가져와 AI 학습용으로 전처리하는 스크립트가 포함되어 있습니다.

## 포함 파일
- `image_preprocessing.py` : 전처리 메인 스크립트 (한글 주석 포함)
- `preprocessed_samples/` : 스크립트 실행 시 최대 5장의 샘플 이미지가 여기에 저장됩니다.

## 요구사항
- 크기 조정: 224×224
- 색상 변환: 그레이스케일(선택) 및 정규화(선택)
- 노이즈 제거: Gaussian Blur 적용
- 데이터 증강: 좌우 반전, 회전, 색상 변화
- 이상치 필터: 평균 밝기 기준으로 어두운 이미지 제거; 주요 객체가 너무 작은 이미지 제거

## 사용 방법 (PowerShell 예)
1) 작업 디렉터리로 이동
```powershell
Set-Location C:\coding\vision\ai_vision
```

2) (권장) 가상환경 생성 및 활성화
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) 필요한 패키지 설치
```powershell
pip install --upgrade pip
pip install opencv-python numpy Pillow tqdm
# datasets 라이브러리로 Hugging Face에서 직접 다운로드하려면 아래도 설치
pip install datasets
```

4) 원본 이미지 폴더에 이미지(예: `raw_images/`)를 넣고 스크립트 실행
```powershell
python image_preprocessing.py --input-dir raw_images --output-dir preprocessed --sample-dir preprocessed_samples --size 224 --blur 3 --augment --filter-dark --filter-small-object
```

5) 제출 항목
- `image_preprocessing.py` : 이 스크립트
- `preprocessed_samples/` : 처리된 이미지 5장
- `README.md` : 전처리 과정 설명(이 파일)

## 튜닝 팁
- `--min-brightness` 값을 조정해 어두운 이미지 필터 기준 변경
- `--min-obj-ratio` 값을 조정해 작은 객체 필터 기준 변경
- `--blur` 커널 크기를 키우면 더 강한 노이즈 제거

문의: 실행 중 에러가 발생하면 터미널 출력 내용을 복사해 알려주시면 원인 분석을 도와드리겠습니다.
