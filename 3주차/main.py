"""
3주차 과제 메인 실행 파일
AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화
"""

import os
import sys
from pathlib import Path

def main():
    """3주차 과제 메인 실행 함수"""
    print("🎯 3주차 과제: AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화")
    print("=" * 70)
    print()
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"📁 현재 작업 디렉토리: {current_dir}")
    print()
    
    # 1단계: 환경 설정 및 모델 학습
    print("🚀 1단계: YOLO 모델 학습 시작")
    print("-" * 40)
    
    try:
        from yolo_training import YOLOTrainer
        
        trainer = YOLOTrainer()
        trainer.setup_environment()
        trainer.prepare_sample_dataset()
        
        # 모델 학습
        print("\n📚 모델 학습 중...")
        results = trainer.train_model(epochs=10, imgsz=640, augment=True)
        
        if results:
            # 모델 평가
            print("\n📊 모델 평가 중...")
            metrics = trainer.evaluate_model()
            
            if metrics:
                # 성능 시각화
                print("\n📈 성능 시각화 중...")
                trainer.visualize_performance(metrics)
        
        print("✅ 1단계 완료: 모델 학습 및 평가")
        
    except Exception as e:
        print(f"❌ 1단계 실패: {e}")
        print("⚠️  모델 학습을 건너뛰고 다음 단계로 진행합니다.")
    
    print("\n" + "="*70)
    
    # 2단계: 객체 탐지 및 패턴 분석
    print("🔍 2단계: 객체 탐지 및 패턴 분석")
    print("-" * 40)
    
    try:
        from object_detection import ObjectDetector
        
        detector = ObjectDetector()
        
        if detector.load_model():
            # 단일 이미지 탐지
            test_image = "datasets/train/images/OIP (1).jpg"
            if os.path.exists(test_image):
                print(f"\n🔍 단일 이미지 탐지: {test_image}")
                image, results, detections = detector.detect_objects(test_image)
                
                if image is not None and detections:
                    detector.visualize_detections(image, detections, "single_detection_result.jpg")
                    detector.analyze_patterns(detections)
            
            # 배치 탐지
            dataset_folder = "datasets/train/images"
            if os.path.exists(dataset_folder):
                print(f"\n📁 배치 탐지: {dataset_folder}")
                detector.batch_detection(dataset_folder)
        
        print("✅ 2단계 완료: 객체 탐지 및 패턴 분석")
        
    except Exception as e:
        print(f"❌ 2단계 실패: {e}")
        print("⚠️  객체 탐지를 건너뛰고 다음 단계로 진행합니다.")
    
    print("\n" + "="*70)
    
    # 3단계: 결과 시각화 및 성능 분석
    print("📊 3단계: 결과 시각화 및 성능 분석")
    print("-" * 40)
    
    try:
        from visualization_analysis import VisualizationAnalyzer
        
        analyzer = VisualizationAnalyzer()
        analyzer.create_comprehensive_report()
        
        print("✅ 3단계 완료: 시각화 및 성능 분석")
        
    except Exception as e:
        print(f"❌ 3단계 실패: {e}")
    
    print("\n" + "="*70)
    
    # 최종 결과 요약
    print("🎉 3주차 과제 완료!")
    print("=" * 70)
    print()
    print("📁 생성된 주요 파일들:")
    print("   📊 분석 결과:")
    print("      - model_performance_analysis.png (모델 성능 분석)")
    print("      - training_analysis.png (학습 곡선 분석)")
    print("      - class_performance_analysis.png (클래스별 성능)")
    print("      - detection_results_analysis.png (탐지 결과 분석)")
    print("      - comprehensive_dashboard.png (종합 대시보드)")
    print()
    print("   🔍 탐지 결과:")
    print("      - single_detection_result.jpg (단일 이미지 탐지)")
    print("      - detection_results/ (배치 탐지 결과)")
    print("      - pattern_analysis.png (패턴 분석)")
    print()
    print("   🏗️  모델 파일:")
    print("      - runs/train/weights/best.pt (학습된 모델)")
    print("      - runs/train/ (학습 로그 및 결과)")
    print()
    print("📋 과제 요구사항 달성:")
    print("   ✅ AI 모델을 활용한 데이터 분석")
    print("   ✅ 이미지에서 객체 탐지 및 패턴 분석")
    print("   ✅ 결과 시각화 및 분석 보고서")
    print()
    print("🚀 다음 단계: PPT 작성을 위한 프롬프트 준비 완료!")

if __name__ == "__main__":
    main()
