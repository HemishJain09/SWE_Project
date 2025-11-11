import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- Singleton Class to load the model only once ---
class FinBERTModel:
    _instance = None

    def __new__(cls, model_name="ProsusAI/finbert"):
        if cls._instance is None:
            print("Initializing and loading FinBERT model...")
            cls._instance = super(FinBERTModel, cls).__new__(cls)
            cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {cls.device}")

            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls.model.to(cls.device)
            cls.model.eval()

            cls.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            print("Model loaded successfully!")
        return cls._instance

def predict(texts, batch_size=16):
    """
    Predict sentiment for a list of texts using the singleton FinBERT model.
    """
    model_instance = FinBERTModel()
    tokenizer = model_instance.tokenizer
    model = model_instance.model
    device = model_instance.device
    label_map = model_instance.label_map

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Verifying with FinBERT"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
            ).to(device)
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_preds.cpu().numpy())

    return [label_map[pred] for pred in predictions]

def run_verification(data_df: pd.DataFrame):
    """
    Main function to run the verification workflow on a DataFrame.

    Args:
        data_df (pd.DataFrame): DataFrame with 'review' and 'sentiment' columns.

    Returns:
        dict: A dictionary containing the verification report metrics.
    """
    texts = data_df['review'].tolist()
    y_true = data_df['sentiment'].tolist()

    # Get predictions from the FinBERT model
    y_pred = predict(texts)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=['positive', 'negative', 'neutral'], zero_division=0
    )

    total = len(y_true)
    correct = int(accuracy * total)

    report = {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision_macro * 100, 2),
        'recall': round(recall_macro * 100, 2),
        'f1_score': round(f1_macro * 100, 2),
        'total_cases': total,
        'correctly_classified': correct
    }
    
    print(f"Verification complete. Report: {report}")
    return report
