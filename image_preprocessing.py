import os
import numpy as np
import cv2
from PIL import Image
from datasets import load_dataset
import torch
import torchvision.transforms as T

# --- 설정값 ---
OUTPUT_DIR = "preprocessed_samples"
NUM_SAMPLES_TO_SAVE = 5
IMAGE_SIZE = (224, 224)

# 심화 문제: 이상치 탐지 기준
BRIGHTNESS_THRESHOLD = 60  # 평균 밝기 이하면 너무 어두운 이미지로 간주
OBJECT_AREA_RATIO_THRESHOLD = 0.1  # 전체 이미지 면적의 10% 이하면 객체가 너무 작은 것으로 간주

# --- 유틸리티 함수 ---
def is_too_dark(pil_image: Image.Image, threshold: int) -> bool:
    """이미지의 평균 밝기를 계산하여 너무 어두운지 확인합니다."""
    # 이미지를 NumPy 배열로 변환 (밝기 계산을 위해 흑백으로)
    grayscale_image = np.array(pil_image.convert("L"))
    mean_brightness = np.mean(grayscale_image)
    return mean_brightness < threshold

def has_small_object(pil_image: Image.Image, threshold: float) -> bool:
    """이미지에서 가장 큰 객체의 면적 비율을 계산하여 너무 작은지 확인합니다."""
    # PIL 이미지를 OpenCV 형식(NumPy 배열)으로 변환
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # 객체 탐지를 위해 흑백으로 변환 후 이진화
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # Otsu의 이진화 알고리즘을 사용하여 최적의 임계값 자동 결정
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 윤곽선(객체) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return True  # 객체를 찾지 못하면 이상치로 간주
    
    # 가장 큰 윤곽선(객체)의 면적 계산
    largest_contour = max(contours, key=cv2.contourArea)
    object_area = cv2.contourArea(largest_contour)
    
    # 전체 이미지 대비 객체 면적 비율 계산
    image_area = pil_image.width * pil_image.height
    ratio = object_area / image_area
    
    return ratio < threshold

def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    """정규화된 텐서 이미지를 저장 가능한 PIL 이미지로 변환합니다."""
    # 역정규화 (Normalize의 역연산)
    # Normalize(mean=[0.5], std=[0.5])는 (pixel - 0.5) / 0.5 연산을 수행
    # 역연산은 (tensor * 0.5) + 0.5
    inversed_tensor = tensor_image * 0.5 + 0.5
    # 텐서의 값을 [0, 1] 범위로 클램핑
    inversed_tensor = torch.clamp(inversed_tensor, 0, 1)
    
    # 텐서를 PIL 이미지로 변환
    return T.ToPILImage()(inversed_tensor)

def main():
    """메인 전처리 파이프라인을 실행합니다."""
    print("🚀 이미지 전처리 파이프라인을 시작합니다.")

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 저장 디렉토리: '{OUTPUT_DIR}/'")

    # Hugging Face 데이터셋 로드 (대용량이므로 스트리밍 모드 사용)
    # streaming=True를 사용하면 전체 데이터셋을 다운로드하지 않고 하나씩 가져옵니다.
    dataset = load_dataset("food101", split="train", streaming=True)
    
    # 기본 문제: 전처리 및 증강 파이프라인 정의
    # torchvision.transforms를 사용하여 순차적인 변환을 정의합니다.
    preprocess_pipeline = T.Compose([
        # 1. 크기 조정
        T.Resize(IMAGE_SIZE),
        
        # 2. 데이터 증강 (Data Augmentation)
        T.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
        T.RandomRotation(30),           # -30도 ~ +30도 사이로 랜덤 회전
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 색상 특성 랜덤 변경
        
        # 3. 색상 변환 및 노이즈 제거
        T.Grayscale(),                                  # 흑백으로 변환
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)), # 가우시안 블러 필터로 노이즈 제거
        
        # 4. 텐서 변환 및 정규화
        T.ToTensor(),                   # PIL 이미지를 PyTorch 텐서로 변환 ([0, 255] -> [0, 1])
        T.Normalize(mean=[0.5], std=[0.5]) # 픽셀 값을 [-1, 1] 범위로 정규화 (흑백이므로 채널 1개)
    ])
    
    saved_count = 0
    processed_count = 0
    
    print("\n데이터셋을 순회하며 이미지 처리를 시작합니다...")
    for item in dataset:
        if saved_count >= NUM_SAMPLES_TO_SAVE:
            print(f"\n목표한 샘플 {NUM_SAMPLES_TO_SAVE}장을 모두 저장했습니다.")
            break

        processed_count += 1
        original_pil = item['image'].convert("RGB") # RGBA 등 다른 형식을 RGB로 통일

        # 심화 문제: 이상치 탐지 및 필터링
        if is_too_dark(original_pil, BRIGHTNESS_THRESHOLD):
            print(f"  - {processed_count}번째 이미지: 너무 어두워 건너뜁니다.")
            continue
            
        if has_small_object(original_pil, OBJECT_AREA_RATIO_THRESHOLD):
            print(f"  - {processed_count}번째 이미지: 객체가 너무 작아 건너뜁니다.")
            continue
        
        # 기본 문제: 전처리 실행
        processed_tensor = preprocess_pipeline(original_pil)
        
        # 처리된 텐서를 저장 가능한 PIL 이미지로 변환
        output_image = tensor_to_pil(processed_tensor)

        # 이미지 저장
        save_path = os.path.join(OUTPUT_DIR, f"sample_{saved_count + 1}.png")
        output_image.save(save_path)
        print(f"  ✔ {processed_count}번째 이미지 처리 완료 -> '{save_path}' 저장 ({saved_count + 1}/{NUM_SAMPLES_TO_SAVE})")
        
        saved_count += 1

    print("\n✅ 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()