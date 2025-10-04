"""
FinBERT Model Verification Script
Uses ProsusAI/finbert model from Hugging Face to verify financial sentiment classification.

Features:
- 🤖 Loads FinBERT from Hugging Face automatically
- 📊 Splits CSV data 50/50 for prediction vs verification
- 📈 Calculates comprehensive metrics (accuracy, precision, recall, F1)
- 📉 Generates confusion matrix and confidence distribution plots
- 💾 Saves detailed report, visualizations, and predictions
- ⚡ GPU support for faster inference
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FinBERTVerification:
    """FinBERT model verification class for sentiment analysis."""

    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialize FinBERT model and tokenizer."""
        print(f"📥 Loading {model_name} model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT label mapping
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.reverse_label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        print("✅ Model loaded successfully!\n")

    def predict(self, texts, batch_size=16):
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for inference

        Returns:
            predictions: List of predicted labels
            confidences: List of confidence scores
        """
        predictions = []
        confidences = []

        print(f"🔮 Predicting sentiments for {len(texts)} texts...")

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Predict
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Get predictions and confidence scores
                batch_confidences, batch_preds = torch.max(probs, dim=-1)

                predictions.extend(batch_preds.cpu().numpy())
                confidences.extend(batch_confidences.cpu().numpy())

        # Convert numeric predictions to labels
        predicted_labels = [self.label_map[pred] for pred in predictions]

        return predicted_labels, confidences

    def load_and_split_data(self, csv_path, text_column, label_column, test_size=0.5, random_state=42):
        """
        Load CSV data and split into train/test sets.

        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            test_size: Proportion for verification set (default 0.5 for 50/50 split)
            random_state: Random seed for reproducibility

        Returns:
            train_df, test_df: Split dataframes
        """
        print(f"📂 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        print(f"📊 Dataset size: {len(df)} samples")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🏷️  Label distribution:\n{df[label_column].value_counts()}\n")

        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_column]
        )

        print(f"✂️  Split completed:")
        print(f"   Training set: {len(train_df)} samples")
        print(f"   Verification set: {len(test_df)} samples\n")

        return train_df, test_df

    def calculate_metrics(self, y_true, y_pred, confidences):
        """
        Calculate comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidence scores

        Returns:
            metrics_dict: Dictionary containing all metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=['positive', 'negative', 'neutral']
        )

        # Macro-averaged metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])

        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }

        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'confidence_stats': confidence_stats,
            'classification_report': classification_report(
                y_true, y_pred, labels=['positive', 'negative', 'neutral']
            )
        }

        return metrics

    def generate_visualizations(self, metrics, confidences, y_true, output_path='verification_plots.png'):
        """
        Generate confusion matrix and confidence distribution plots.

        Args:
            metrics: Metrics dictionary
            confidences: List of confidence scores
            y_true: True labels
            output_path: Path to save plots
        """
        print(f"📊 Generating visualizations...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['positive', 'negative', 'neutral'],
            yticklabels=['positive', 'negative', 'neutral'],
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)

        # Confidence Distribution
        axes[1].hist(confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(
            np.mean(confidences),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {np.mean(confidences):.3f}'
        )
        axes[1].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Confidence Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved to {output_path}\n")
        plt.close()

    def generate_report(self, metrics, output_path='verification_report.txt'):
        """
        Generate comprehensive text report.

        Args:
            metrics: Metrics dictionary
            output_path: Path to save report
        """
        print(f"📝 Generating verification report...")

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FinBERT MODEL VERIFICATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: ProsusAI/finbert\n")
            f.write("=" * 80 + "\n\n")

            # Overall Metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro):    {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro):  {metrics['f1_macro']:.4f}\n")
            f.write("\n")

            # Per-Class Performance
            f.write("PER-CLASS PERFORMANCE BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            classes = ['positive', 'negative', 'neutral']
            f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
            f.write("-" * 80 + "\n")
            for i, cls in enumerate(classes):
                f.write(f"{cls:<12} {metrics['precision_per_class'][i]:<12.4f} "
                       f"{metrics['recall_per_class'][i]:<12.4f} "
                       f"{metrics['f1_per_class'][i]:<12.4f} "
                       f"{int(metrics['support_per_class'][i]):<12}\n")
            f.write("\n")

            # Confusion Matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'':>12} {'positive':>12} {'negative':>12} {'neutral':>12}\n")
            cm = metrics['confusion_matrix']
            for i, cls in enumerate(classes):
                f.write(f"{cls:>12} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}\n")
            f.write("\n")

            # Confidence Statistics
            f.write("PREDICTION CONFIDENCE STATISTICS\n")
            f.write("-" * 80 + "\n")
            cs = metrics['confidence_stats']
            f.write(f"Mean:   {cs['mean']:.4f}\n")
            f.write(f"Std:    {cs['std']:.4f}\n")
            f.write(f"Min:    {cs['min']:.4f}\n")
            f.write(f"Max:    {cs['max']:.4f}\n")
            f.write(f"Median: {cs['median']:.4f}\n")
            f.write("\n")

            # Detailed Classification Report
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 80 + "\n")
            f.write(metrics['classification_report'])
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"✅ Report saved to {output_path}\n")

    def save_results(self, test_df, predictions, confidences, text_column, label_column,
                    output_path='verification_results.csv'):
        """
        Save predictions and results to CSV.

        Args:
            test_df: Test dataframe
            predictions: List of predicted labels
            confidences: List of confidence scores
            text_column: Name of text column
            label_column: Name of label column
            output_path: Path to save results
        """
        print(f"💾 Saving results to CSV...")

        results_df = test_df.copy()
        results_df['predicted_label'] = predictions
        results_df['confidence'] = confidences
        results_df['correct'] = results_df[label_column] == results_df['predicted_label']

        # Reorder columns for better readability
        cols = [text_column, label_column, 'predicted_label', 'confidence', 'correct']
        results_df = results_df[cols]

        results_df.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}\n")

        # Print some statistics
        correct_preds = results_df['correct'].sum()
        total_preds = len(results_df)
        print(f"📊 Prediction Summary:")
        print(f"   Correct: {correct_preds}/{total_preds} ({100*correct_preds/total_preds:.2f}%)")
        print(f"   Incorrect: {total_preds - correct_preds}/{total_preds} ({100*(total_preds-correct_preds)/total_preds:.2f}%)\n")


def predict_custom_text(verifier):
    """
    Interactive mode for custom text sentiment prediction.

    Args:
        verifier: FinBERTVerification instance
    """
    print("\n" + "=" * 80)
    print("🔮 CUSTOM SENTIMENT PREDICTION MODE")
    print("=" * 80)
    print("Enter text to analyze sentiment (or 'quit' to exit)")
    print("-" * 80 + "\n")

    while True:
        user_input = input("📝 Enter your text: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting custom prediction mode...\n")
            break

        if not user_input:
            print("⚠️  Please enter some text.\n")
            continue

        # Predict sentiment
        predictions, confidences = verifier.predict([user_input], batch_size=1)
        predicted_label = predictions[0]
        confidence = confidences[0]

        # Display result
        print("\n" + "-" * 80)
        print("📊 SENTIMENT ANALYSIS RESULT")
        print("-" * 80)
        print(f"Sentiment: {predicted_label.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("-" * 80 + "\n")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="FinBERT Sentiment Verification")
    parser.add_argument('--mode', type=str, default='verify',
                       choices=['verify', 'custom'],
                       help='Mode: verify (CSV verification) or custom (interactive text input)')
    args = parser.parse_args()

    # Initialize FinBERT verification
    verifier = FinBERTVerification()

    if args.mode == 'custom':
        # Custom text prediction mode
        predict_custom_text(verifier)
    else:
        # Configuration for CSV verification
        CSV_PATH = 'sample_financial_test_cases.csv'
        TEXT_COLUMN = 'review'
        LABEL_COLUMN = 'sentiment'

        # Load and split data
        train_df, test_df = verifier.load_and_split_data(
            csv_path=CSV_PATH,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.5
        )

        # Get predictions on verification set
        texts = test_df[TEXT_COLUMN].tolist()
        predictions, confidences = verifier.predict(texts)

        # Calculate metrics
        y_true = test_df[LABEL_COLUMN].tolist()
        metrics = verifier.calculate_metrics(y_true, predictions, confidences)

        # Generate outputs
        verifier.generate_visualizations(metrics, confidences, y_true, 'verification_plots.png')
        verifier.generate_report(metrics, 'verification_report.txt')
        verifier.save_results(
            test_df, predictions, confidences,
            TEXT_COLUMN, LABEL_COLUMN,
            'verification_results.csv'
        )

        # Print summary
        print("=" * 80)
        print("🎉 VERIFICATION COMPLETE!")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"📉 Mean Confidence: {metrics['confidence_stats']['mean']:.4f}")
        print("\n📁 Output Files:")
        print("   ✓ verification_report.txt")
        print("   ✓ verification_plots.png")
        print("   ✓ verification_results.csv")
        print("=" * 80)


if __name__ == "__main__":
    main()
