from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyEvaluator:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        
    def evaluate(self, test_data, true_labels=None):
        """Đánh giá model trên test data"""
        anomalies, errors = self.model.detect_anomalies(test_data)
        
        # Calculate metrics
        results = {
            'total_sequences': len(test_data),
            'detected_anomalies': np.sum(anomalies),
            'anomaly_rate': np.mean(anomalies),
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'threshold': self.threshold
        }
        
        # If we have true labels, calculate supervised metrics
        if true_labels is not None:
            precision = precision_score(true_labels, anomalies)
            recall = recall_score(true_labels, anomalies)
            f1 = f1_score(true_labels, anomalies)
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, anomalies)
            results['confusion_matrix'] = cm
        
        return results, anomalies, errors
    
    def plot_evaluation_results(self, results, errors, anomalies):
        """Vẽ các biểu đồ đánh giá"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reconstruction errors distribution
        axes[0, 0].hist(errors, bins=50, alpha=0.7)
        axes[0, 0].axvline(self.threshold, color='red', linestyle='--', 
                         label=f'Threshold: {self.threshold:.4f}')
        axes[0, 0].set_title('Distribution of Reconstruction Errors')
        axes[0, 0].set_xlabel('Reconstruction Error (MSE)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Errors with anomalies highlighted
        axes[0, 1].plot(errors, 'b-', alpha=0.7)
        anomaly_indices = np.where(anomalies)[0]
        axes[0, 1].scatter(anomaly_indices, errors[anomalies], 
                         color='red', s=50, label='Detected Anomalies')
        axes[0, 1].axhline(self.threshold, color='red', linestyle='--', 
                         label=f'Threshold')
        axes[0, 1].set_title('Reconstruction Errors Over Time')
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('Reconstruction Error')
        axes[0, 1].legend()
        
        # Metrics summary
        if 'precision' in results:
            metrics_text = f"""
            Precision: {results['precision']:.3f}
            Recall: {results['recall']:.3f}
            F1-Score: {results['f1_score']:.3f}
            
            Detected Anomalies: {results['detected_anomalies']}/{results['total_sequences']}
            Anomaly Rate: {results['anomaly_rate']:.1%}"""
        else:
            metrics_text = f"""
            Detected Anomalies: {results['detected_anomalies']}/{results['total_sequences']}
            Anomaly Rate: {results['anomaly_rate']:.1%}
            Mean Error: {results['mean_error']:.4f}
            Max Error: {results['max_error']:.4f}"""
            
        axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center', fontfamily='monospace')
        axes[1, 0].set_title('Evaluation Metrics')
        axes[1, 0].axis('off')
        
        # Confusion matrix if available
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Anomaly'], 
                       yticklabels=['Normal', 'Anomaly'], ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('models/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

def print_evaluation_summary(results):
    """In tóm tắt kết quả đánh giá"""
    print("\n" + "="*50)
    print("ANOMALY DETECTION EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total sequences analyzed: {results['total_sequences']}")
    print(f"Detected anomalies: {results['detected_anomalies']}")
    print(f"Anomaly detection rate: {results['anomaly_rate']:.1%}")
    print(f"Anomaly threshold: {results['threshold']:.4f}")
    print(f"Mean reconstruction error: {results['mean_error']:.4f}")
    print(f"Maximum error: {results['max_error']:.4f}")
    
    if 'precision' in results:
        print(f"\nSupervised Metrics:")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1-Score: {results['f1_score']:.3f}")
    
    print("="*50)
