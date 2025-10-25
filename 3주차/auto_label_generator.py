"""
자동 라벨 생성 도구
YOLO 모델을 사용하여 이미지에서 객체를 탐지하고 라벨 파일을 자동 생성
"""

import os
import cv2
from ultralytics import YOLO

def auto_generate_labels():
    """자동 라벨 생성"""
    print("🤖 자동 라벨 생성 시작...")
    
    # 사전 훈련된 YOLO 모델 로드
    model = YOLO("yolov8n.pt")
    
    # 디렉토리 생성
    os.makedirs("datasets/train/labels", exist_ok=True)
    os.makedirs("datasets/valid/labels", exist_ok=True)
    
    # 이미지 파일들 (실제 파일명으로 수정)
    train_images = [
        "datasets/train/images/OIP (1).jpg",
        "datasets/train/images/OIP (2).jpg", 
        "datasets/train/images/OIP (3).jpg",
        "datasets/train/images/OIP (4).jpg"
    ]
    
    valid_images = [
        "datasets/train/images/OIP (5).jpg"  # valid 폴더가 비어있으므로 train에서 하나 사용
    ]
    
    # COCO 클래스 매핑 (person, car, dog)
    class_mapping = {
        0: 0,   # person -> person (class_id = 0)
        2: 1,   # car -> car (class_id = 1) 
        16: 2   # dog -> dog (class_id = 2)
    }
    
    def process_images(image_list, output_dir):
        """이미지들을 처리하여 라벨 파일 생성"""
        for image_path in image_list:
            if os.path.exists(image_path):
                print(f"📸 처리 중: {image_path}")
                
                # 객체 탐지
                results = model(image_path)
                
                # 라벨 파일 생성
                image_name = os.path.basename(image_path)
                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(output_dir, label_name)
                
                with open(label_path, "w") as f:
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # person, car, dog만 필터링
                            if class_id in class_mapping and confidence > 0.5:
                                # 좌표 변환
                                x1, y1, x2, y2 = box.xyxy[0]
                                img = cv2.imread(image_path)
                                h, w = img.shape[:2]
                                
                                # 정규화된 좌표 계산
                                x_center = (x1 + x2) / 2 / w
                                y_center = (y1 + y2) / 2 / h
                                width = (x2 - x1) / w
                                height = (y2 - y1) / h
                                
                                # 라벨 파일에 쓰기
                                mapped_class = class_mapping[class_id]
                                f.write(f"{mapped_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                print(f"✅ 라벨 생성: {label_path}")
    
    # 학습용 이미지 처리
    process_images(train_images, "datasets/train/labels")
    
    # 검증용 이미지 처리
    process_images(valid_images, "datasets/valid/labels")
    
    print("\n🎉 자동 라벨 생성 완료!")
    print("📋 생성된 라벨 파일들:")
    print("   - datasets/train/labels/")
    print("   - datasets/valid/labels/")

def create_sample_labels():
    """샘플 라벨 파일 생성 (수동 라벨 생성용)"""
    print("📝 샘플 라벨 파일 생성 중...")
    
    # 디렉토리 생성
    os.makedirs("datasets/train/labels", exist_ok=True)
    os.makedirs("datasets/valid/labels", exist_ok=True)
    
    # 샘플 라벨 파일들 (실제 파일명으로 수정)
    sample_labels = {
        "OIP (1).txt": "0 0.5 0.5 0.3 0.4",  # person
        "OIP (2).txt": "1 0.5 0.5 0.6 0.3",  # car
        "OIP (3).txt": "2 0.5 0.5 0.4 0.5",  # dog
        "OIP (4).txt": "0 0.5 0.5 0.3 0.4",  # person
        "OIP (5).txt": "1 0.5 0.5 0.6 0.3"   # car
    }
    
    # 학습용 라벨 파일 생성
    for filename, content in sample_labels.items():
        if filename in ["OIP (1).txt", "OIP (2).txt", "OIP (3).txt", "OIP (4).txt"]:
            with open(f"datasets/train/labels/{filename}", "w") as f:
                f.write(content)
            print(f"✅ 생성됨: datasets/train/labels/{filename}")
        else:
            with open(f"datasets/valid/labels/{filename}", "w") as f:
                f.write(content)
            print(f"✅ 생성됨: datasets/valid/labels/{filename}")
    
    print("\n🎉 샘플 라벨 파일 생성 완료!")

def main():
    """메인 실행 함수"""
    print("🏷️  라벨 파일 생성 도구")
    print("=" * 40)
    
    print("1. 자동 라벨 생성 (YOLO 모델 사용)")
    print("2. 샘플 라벨 파일 생성 (수동 라벨 생성용)")
    
    choice = input("\n선택하세요 (1 또는 2): ")
    
    if choice == "1":
        auto_generate_labels()
    elif choice == "2":
        create_sample_labels()
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
