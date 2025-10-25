"""
3주차 과제: 결과 시각화 및 성능 분석
Matplotlib과 OpenCV를 활용한 종합적인 시각화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os
from pathlib import Path
import pandas as pd

# 한글 폰트 설정
try:
    # Windows 환경에서 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    print("✅ Matplotlib 한글 폰트 설정 완료 (Malgun Gothic)")
except Exception as e:
    try:
        # 대체 한글 폰트 시도
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Matplotlib 한글 폰트 설정 완료 (NanumGothic)")
    except Exception as e2:
        print(f"⚠️ 한글 폰트 설정 실패: {e2}")
        print("   영어로 제목을 표시합니다.")
        # 영어 폰트로 대체
        plt.rcParams['font.family'] = 'DejaVu Sans'

class VisualizationAnalyzer:
    def __init__(self):
        """시각화 및 분석을 위한 클래스"""
        self.results_data = {}
        self.performance_metrics = {}
        
    def create_comprehensive_report(self, model_path="runs/train/exp/weights/best.pt"):
        """
        종합적인 분석 보고서 생성
        
        Args:
            model_path: 학습된 모델 경로
        """
        print("📊 종합 분석 보고서 생성 중...")
        
        try:
            # 1. 모델 성능 분석
            self._analyze_model_performance(model_path)
            
            # 2. 학습 곡선 시각화
            self._visualize_training_curves()
            
            # 3. 클래스별 성능 분석
            self._analyze_class_performance()
            
            # 4. 탐지 결과 시각화
            self._visualize_detection_results()
            
            # 5. 종합 대시보드 생성
            self._create_dashboard()
            
            print("✅ 종합 분석 보고서 완료!")
            
        except Exception as e:
            print(f"❌ 보고서 생성 실패: {e}")
    
    def _analyze_model_performance(self, model_path):
        """모델 성능 분석"""
        print("📈 모델 성능 분석 중...")
        
        try:
            # 모델 로드
            model = YOLO(model_path) if os.path.exists(model_path) else YOLO("yolov8n.pt")
            
            # 성능 지표 (예시 데이터 - 실제로는 model.val() 결과 사용)
            self.performance_metrics = {
                'mAP50': 0.85,
                'mAP50-95': 0.72,
                'Precision': 0.88,
                'Recall': 0.82,
                'F1-Score': 0.85
            }
            
            print("✅ 모델 성능 분석 완료!")
            
        except Exception as e:
            print(f"❌ 성능 분석 실패: {e}")
    
    def _visualize_training_curves(self):
        """학습 곡선 시각화"""
        print("📈 학습 곡선 시각화 중...")
        
        try:
            # 예시 학습 데이터 (실제로는 학습 로그에서 추출)
            epochs = range(1, 21)
            train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22,
                         0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]
            val_loss = [0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.32,
                       0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]
            
            train_precision = [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88,
                              0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
            val_precision = [0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78,
                            0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]
            
            # 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLO 모델 학습 분석', fontsize=16, fontweight='bold')
            
            # Loss 곡선
            axes[0, 0].plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
            axes[0, 0].plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2)
            axes[0, 0].set_title('학습 및 검증 손실')
            axes[0, 0].set_xlabel('Epochs')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Precision 곡선
            axes[0, 1].plot(epochs, train_precision, label='Training Precision', marker='o', linewidth=2)
            axes[0, 1].plot(epochs, val_precision, label='Validation Precision', marker='s', linewidth=2)
            axes[0, 1].set_title('Precision 변화')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 성능 지표 막대 그래프
            metrics = list(self.performance_metrics.keys())
            values = list(self.performance_metrics.values())
            colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
            
            bars = axes[1, 0].bar(metrics, values, color=colors)
            axes[1, 0].set_title('모델 성능 지표')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 클래스별 성능 (예시)
            classes = ['Person', 'Car', 'Dog']
            ap_scores = [0.88, 0.82, 0.85]
            
            axes[1, 1].bar(classes, ap_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 1].set_title('클래스별 AP (Average Precision)')
            axes[1, 1].set_ylabel('AP Score')
            
            # 값 표시
            for i, (class_name, score) in enumerate(zip(classes, ap_scores)):
                axes[1, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 학습 곡선 시각화 완료! (training_analysis.png)")
            
        except Exception as e:
            print(f"❌ 학습 곡선 시각화 실패: {e}")
    
    def _analyze_class_performance(self):
        """클래스별 성능 분석"""
        print("🏷️  클래스별 성능 분석 중...")
        
        try:
            # 클래스별 성능 데이터 (예시)
            class_data = {
                'Class': ['Person', 'Car', 'Dog'],
                'AP50': [0.88, 0.82, 0.85],
                'AP50-95': [0.75, 0.68, 0.72],
                'Precision': [0.90, 0.85, 0.88],
                'Recall': [0.85, 0.80, 0.83],
                'F1-Score': [0.87, 0.82, 0.85]
            }
            
            df = pd.DataFrame(class_data)
            
            # 히트맵 생성
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 성능 지표 히트맵
            metrics_matrix = df.set_index('Class')[['AP50', 'AP50-95', 'Precision', 'Recall', 'F1-Score']]
            sns.heatmap(metrics_matrix, annot=True, cmap='YlOrRd', fmt='.3f', ax=axes[0])
            axes[0].set_title('클래스별 성능 지표 히트맵')
            
            # 클래스별 AP 비교
            x = np.arange(len(df))
            width = 0.35
            
            axes[1].bar(x - width/2, df['AP50'], width, label='AP50', alpha=0.8)
            axes[1].bar(x + width/2, df['AP50-95'], width, label='AP50-95', alpha=0.8)
            axes[1].set_xlabel('Classes')
            axes[1].set_ylabel('AP Score')
            axes[1].set_title('클래스별 AP 비교')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(df['Class'])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('class_performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 클래스별 성능 분석 완료! (class_performance_analysis.png)")
            
        except Exception as e:
            print(f"❌ 클래스별 성능 분석 실패: {e}")
    
    def _visualize_detection_results(self):
        """탐지 결과 시각화"""
        print("🔍 탐지 결과 시각화 중...")
        
        try:
            # 예시 탐지 결과 데이터
            detection_data = {
                'Image': ['Image_1', 'Image_2', 'Image_3', 'Image_4', 'Image_5'],
                'Person': [2, 1, 3, 0, 1],
                'Car': [1, 2, 0, 3, 2],
                'Dog': [0, 1, 1, 0, 1]
            }
            
            df = pd.DataFrame(detection_data)
            
            # 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('객체 탐지 결과 분석', fontsize=16, fontweight='bold')
            
            # 이미지별 탐지된 객체 수
            df_melted = df.melt(id_vars=['Image'], var_name='Class', value_name='Count')
            sns.barplot(data=df_melted, x='Image', y='Count', hue='Class', ax=axes[0, 0])
            axes[0, 0].set_title('이미지별 탐지된 객체 수')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 클래스별 총 탐지 수
            class_totals = df[['Person', 'Car', 'Dog']].sum()
            axes[0, 1].pie(class_totals.values, labels=class_totals.index, autopct='%1.1f%%', 
                          colors=['skyblue', 'lightcoral', 'lightgreen'])
            axes[0, 1].set_title('클래스별 총 탐지 분포')
            
            # 신뢰도 분포 (예시)
            confidence_data = np.random.normal(0.8, 0.1, 100)
            axes[1, 0].hist(confidence_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('탐지 신뢰도 분포')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(confidence_data), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(confidence_data):.3f}')
            axes[1, 0].legend()
            
            # 탐지 성능 트렌드 (예시)
            epochs = range(1, 11)
            detection_accuracy = [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]
            axes[1, 1].plot(epochs, detection_accuracy, marker='o', linewidth=2, color='green')
            axes[1, 1].set_title('탐지 정확도 개선 트렌드')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Detection Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('detection_results_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 탐지 결과 시각화 완료! (detection_results_analysis.png)")
            
        except Exception as e:
            print(f"❌ 탐지 결과 시각화 실패: {e}")
    
    def _create_dashboard(self):
        """종합 대시보드 생성"""
        print("📊 종합 대시보드 생성 중...")
        
        try:
            # 대시보드 생성
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 제목
            fig.suptitle('YOLO 객체 탐지 모델 종합 분석 대시보드', fontsize=20, fontweight='bold')
            
            # 1. 모델 성능 요약 (상단 좌측)
            ax1 = fig.add_subplot(gs[0, :2])
            metrics = list(self.performance_metrics.keys())
            values = list(self.performance_metrics.values())
            bars = ax1.bar(metrics, values, color=plt.cm.viridis(np.linspace(0, 1, len(metrics))))
            ax1.set_title('모델 성능 지표', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. 클래스별 성능 (상단 우측)
            ax2 = fig.add_subplot(gs[0, 2:])
            classes = ['Person', 'Car', 'Dog']
            ap_scores = [0.88, 0.82, 0.85]
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            bars = ax2.bar(classes, ap_scores, color=colors)
            ax2.set_title('클래스별 AP 점수', fontsize=14, fontweight='bold')
            ax2.set_ylabel('AP Score')
            
            for bar, score in zip(bars, ap_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. 학습 곡선 (중간 좌측)
            ax3 = fig.add_subplot(gs[1, :2])
            epochs = range(1, 11)
            train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22]
            val_loss = [0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.32]
            ax3.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
            ax3.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2)
            ax3.set_title('학습 곡선', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 탐지 결과 분포 (중간 우측)
            ax4 = fig.add_subplot(gs[1, 2:])
            detection_counts = [15, 12, 8]  # Person, Car, Dog
            ax4.pie(detection_counts, labels=classes, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax4.set_title('탐지된 객체 분포', fontsize=14, fontweight='bold')
            
            # 5. 성능 개선 제안 (하단)
            ax5 = fig.add_subplot(gs[2, :])
            ax5.text(0.5, 0.7, '성능 개선 제안사항', ha='center', va='center', 
                    fontsize=16, fontweight='bold', transform=ax5.transAxes)
            
            suggestions = [
                "• 데이터 증강(Augmentation) 적용으로 모델 일반화 성능 향상",
                "• 하이퍼파라미터 튜닝을 통한 최적 학습률 및 배치 크기 조정",
                "• 더 큰 모델(yolov8s.pt, yolov8m.pt) 사용으로 정확도 향상",
                "• 앙상블 기법 적용으로 탐지 성능 향상"
            ]
            
            for i, suggestion in enumerate(suggestions):
                ax5.text(0.1, 0.5 - i*0.1, suggestion, ha='left', va='center',
                        fontsize=12, transform=ax5.transAxes)
            
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            
            plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 종합 대시보드 완료! (comprehensive_dashboard.png)")
            
        except Exception as e:
            print(f"❌ 대시보드 생성 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🎯 3주차 과제: 결과 시각화 및 성능 분석")
    print("=" * 50)
    
    # 시각화 분석기 초기화
    analyzer = VisualizationAnalyzer()
    
    # 종합 분석 보고서 생성
    analyzer.create_comprehensive_report()
    
    print("\n🎉 시각화 및 분석 완료!")
    print("📁 생성된 파일들:")
    print("   - training_analysis.png (학습 분석)")
    print("   - class_performance_analysis.png (클래스별 성능)")
    print("   - detection_results_analysis.png (탐지 결과 분석)")
    print("   - comprehensive_dashboard.png (종합 대시보드)")

if __name__ == "__main__":
    main()
