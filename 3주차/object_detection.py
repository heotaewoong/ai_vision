"""
3주차 과제: 객체 탐지 및 패턴 분석
OpenCV를 활용한 객체 탐지 및 시각화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from pathlib import Path

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ Matplotlib 한글 폰트 설정 완료 (Malgun Gothic)")
except Exception as e:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Matplotlib 한글 폰트 설정 완료 (NanumGothic)")
    except Exception as e2:
        print(f"⚠️ 한글 폰트 설정 실패: {e2}")
        plt.rcParams['font.family'] = 'DejaVu Sans'

class ObjectDetector:
    def __init__(self, model_path="runs/train/exp/weights/best.pt"):
        """
        객체 탐지를 위한 클래스
        
        Args:
            model_path (str): 학습된 YOLO 모델 경로
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['person', 'car', 'dog']
        
    def load_model(self):
        """사전 훈련된 YOLO 모델 로드"""
        print("📦 사전 훈련된 YOLO 모델 로드 중...")
        
        try:
            # 사전 훈련된 YOLOv8 모델 사용 (COCO 데이터셋으로 훈련됨)
            self.model = YOLO("yolov8n.pt")
            print("✅ 사전 훈련된 YOLOv8 모델 로드 완료!")
            print("   - COCO 데이터셋으로 훈련된 모델 (80개 클래스)")
            print("   - person, car, dog 등 일반적인 객체 탐지 가능")
            print("   - 라벨 파일 없이도 객체 탐지 가능")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
        
        return True
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """
        이미지에서 객체 탐지
        
        Args:
            image_path (str): 이미지 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            tuple: (원본 이미지, 탐지 결과, 바운딩 박스 정보)
        """
        print(f"🔍 객체 탐지 실행: {image_path}")
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
                return None, None, None
            
            # 객체 탐지 실행
            results = self.model(image, conf=confidence_threshold)
            
            # 탐지 결과 처리 (person, car, dog만 필터링)
            target_classes = ['person', 'car', 'dog']
            detections = []
            for result in results:
                for box in result.boxes:
                    # 좌표 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 클래스 정보
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 클래스 이름 (학습된 모델이면 사용자 정의, 아니면 COCO 클래스)
                    if hasattr(result, 'names') and class_id < len(result.names):
                        label = result.names[class_id]
                    else:
                        label = f"class_{class_id}"
                    
                    # person, car, dog만 필터링
                    if label in target_classes:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': class_id,
                            'label': label,
                            'confidence': confidence
                        })
            
            print(f"✅ 탐지 완료: {len(detections)}개 객체 발견")
            return image, results, detections
            
        except Exception as e:
            print(f"❌ 객체 탐지 실패: {e}")
            return None, None, None
    
    def visualize_detections(self, image, detections, save_path=None):
        """
        탐지 결과 시각화
        
        Args:
            image: 원본 이미지
            detections: 탐지 결과 리스트
            save_path: 저장할 파일 경로 (선택사항)
        """
        print("🎨 탐지 결과 시각화 중...")
        
        try:
            # 이미지 복사
            vis_image = image.copy()
            
            # 색상 정의 (BGR 형식)
            colors = [
                (0, 255, 0),    # 녹색
                (255, 0, 0),    # 빨간색
                (0, 0, 255),    # 파란색
                (255, 255, 0),  # 청록색
                (255, 0, 255),  # 자홍색
                (0, 255, 255),  # 노란색
            ]
            
            # 각 탐지된 객체에 대해 바운딩 박스 그리기
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                label = detection['label']
                confidence = detection['confidence']
                color = colors[i % len(colors)]
                
                # 바운딩 박스 그리기
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # 라벨과 신뢰도 텍스트 (더 큰 폰트로)
                text = f"{label}: {confidence:.2f}"
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # 텍스트 배경 (더 넓게)
                cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 15), 
                            (x1 + text_size[0] + 10, y1), color, -1)
                
                # 텍스트 그리기 (흰색으로 더 선명하게)
                cv2.putText(vis_image, text, (x1 + 5, y1 - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # 결과 저장
            if save_path:
                cv2.imwrite(save_path, vis_image)
                print(f"✅ 시각화 결과 저장: {save_path}")
            
            # 화면에 표시
            cv2.imshow("Object Detection Results", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return vis_image
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
            return None
    
    def analyze_patterns(self, detections):
        """
        탐지된 객체들의 패턴 분석
        
        Args:
            detections: 탐지 결과 리스트
        """
        print("📊 패턴 분석 중...")
        
        try:
            if not detections:
                print("⚠️  분석할 탐지 결과가 없습니다.")
                return
            
            # 클래스별 통계
            class_counts = {}
            confidence_scores = []
            
            for detection in detections:
                label = detection['label']
                confidence = detection['confidence']
                
                # 클래스별 카운트
                class_counts[label] = class_counts.get(label, 0) + 1
                confidence_scores.append(confidence)
            
            # 통계 출력
            print("\n📈 탐지 결과 통계:")
            print(f"   - 총 탐지된 객체 수: {len(detections)}")
            print(f"   - 평균 신뢰도: {np.mean(confidence_scores):.3f}")
            print(f"   - 최고 신뢰도: {np.max(confidence_scores):.3f}")
            print(f"   - 최저 신뢰도: {np.min(confidence_scores):.3f}")
            
            print("\n🏷️  클래스별 분포:")
            for class_name, count in class_counts.items():
                percentage = (count / len(detections)) * 100
                print(f"   - {class_name}: {count}개 ({percentage:.1f}%)")
            
            # 패턴 분석 시각화
            self._visualize_pattern_analysis(class_counts, confidence_scores)
            
        except Exception as e:
            print(f"❌ 패턴 분석 실패: {e}")
    
    def _visualize_pattern_analysis(self, class_counts, confidence_scores):
        """패턴 분석 결과 시각화"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('객체 탐지 패턴 분석', fontsize=14, fontweight='bold')
            
            # 클래스별 분포 파이 차트
            if class_counts:
                labels = list(class_counts.keys())
                sizes = list(class_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
                axes[0].set_title('클래스별 분포')
            
            # 신뢰도 분포 히스토그램
            axes[1].hist(confidence_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].set_title('신뢰도 분포')
            axes[1].set_xlabel('신뢰도')
            axes[1].set_ylabel('빈도')
            axes[1].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                          label=f'평균: {np.mean(confidence_scores):.3f}')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('pattern_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 패턴 분석 시각화 완료! (pattern_analysis.png)")
            
        except Exception as e:
            print(f"❌ 패턴 분석 시각화 실패: {e}")
    
    def batch_detection(self, image_folder, output_folder="detection_results"):
        """
        여러 이미지에 대한 배치 객체 탐지
        
        Args:
            image_folder: 이미지 폴더 경로
            output_folder: 결과 저장 폴더
        """
        print(f"📁 배치 객체 탐지 시작: {image_folder}")
        
        try:
            # 출력 폴더 생성
            os.makedirs(output_folder, exist_ok=True)
            
            # 이미지 파일 찾기
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(image_folder).glob(f"*{ext}"))
                image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"❌ {image_folder}에서 이미지 파일을 찾을 수 없습니다.")
                return
            
            print(f"📸 {len(image_files)}개 이미지 발견")
            
            # 각 이미지에 대해 객체 탐지
            all_detections = []
            for i, image_path in enumerate(image_files):
                print(f"\n🔍 처리 중 ({i+1}/{len(image_files)}): {image_path.name}")
                
                image, results, detections = self.detect_objects(str(image_path))
                
                if image is not None and detections:
                    # 시각화 및 저장
                    output_path = os.path.join(output_folder, f"detected_{image_path.name}")
                    self.visualize_detections(image, detections, output_path)
                    
                    # 전체 탐지 결과에 추가
                    all_detections.extend(detections)
            
            # 전체 패턴 분석
            if all_detections:
                print(f"\n📊 전체 배치 분석 ({len(all_detections)}개 탐지)")
                self.analyze_patterns(all_detections)
            
            print(f"✅ 배치 탐지 완료! 결과 저장: {output_folder}")
            
        except Exception as e:
            print(f"❌ 배치 탐지 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🎯 3주차 과제: 객체 탐지 및 패턴 분석")
    print("=" * 50)
    
    # 객체 탐지기 초기화
    detector = ObjectDetector()
    
    # 모델 로드
    if not detector.load_model():
        return
    
    # 실제 이미지 파일들 탐지
    test_images = [
        "datasets/train/images/OIP (1).jpg",
        "datasets/train/images/OIP (2).jpg", 
        "datasets/train/images/OIP (3).jpg",
        "datasets/train/images/OIP (4).jpg",
        "datasets/train/images/OIP (5).jpg"
    ]
    
    print(f"\n🔍 data.yaml 클래스 (person, car, dog)에 맞는 테스트 이미지들 탐지")
    print("=" * 60)
    
    all_detections = []
    for i, test_image in enumerate(test_images):
        if os.path.exists(test_image):
            print(f"\n📸 테스트 이미지 {i+1}: {test_image}")
            image, results, detections = detector.detect_objects(test_image, confidence_threshold=0.3)
            
            if image is not None and detections:
                # 시각화 및 저장
                output_name = f"detected_OIP_{i+1}.jpg"
                detector.visualize_detections(image, detections, f"detection_results/{output_name}")
                print(f"✅ {len(detections)}개 객체 탐지됨")
                
                # 각 탐지 결과 출력
                for j, detection in enumerate(detections):
                    print(f"   {j+1}. {detection['label']}: {detection['confidence']:.3f}")
                
                all_detections.extend(detections)
            else:
                print("⚠️  탐지된 객체가 없습니다.")
        else:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {test_image}")
    
    # 전체 패턴 분석
    if all_detections:
        print(f"\n📊 전체 탐지 결과 분석 ({len(all_detections)}개 탐지)")
        detector.analyze_patterns(all_detections)
    
    # 배치 탐지 (전체 dataset 폴더)
    dataset_folder = "../dataset"
    if os.path.exists(dataset_folder):
        print(f"\n📁 배치 탐지: {dataset_folder}")
        detector.batch_detection(dataset_folder)
    
    print("\n🎉 객체 탐지 및 패턴 분석 완료!")

if __name__ == "__main__":
    main()
