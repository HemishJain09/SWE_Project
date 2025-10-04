"""
FinBERT Dummy Test Script
Tests the FinBERT model with sample financial statements.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


def classify_financial_sentiment(statement, model, tokenizer, device):
    """
    Classify the sentiment (positive, negative, or neutral) of a financial statement.

    Args:
        statement: Financial statement text
        model: FinBERT model
        tokenizer: FinBERT tokenizer
        device: torch device

    Returns:
        sentiment: Predicted sentiment label
        confidence: Confidence score
        all_scores: Dictionary with all sentiment scores
    """
    # Tokenize
    inputs = tokenizer(
        statement,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # FinBERT label mapping
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    # Get prediction
    predicted_idx = probs.argmax()
    sentiment = label_map[predicted_idx]
    confidence = probs[predicted_idx]

    # All scores
    all_scores = {
        'positive': probs[0],
        'negative': probs[1],
        'neutral': probs[2]
    }

    return sentiment, confidence, all_scores


def main():
    """Main test function."""

    print("=" * 80)
    print("FinBERT DUMMY TEST SCRIPT")
    print("=" * 80)
    print()

    # Load model
    print("📥 Loading FinBERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!\n")

    # Test financial statements
    financial_statements = [
        "Apple's quarterly earnings exceeded Wall Street expectations, driven by strong iPhone sales in Asia.",
        "The company reported a significant loss in Q3, with revenue declining by 15% year-over-year.",
        "Tesla announced record deliveries this quarter, surpassing analyst estimates.",
        "The Federal Reserve raised interest rates by 75 basis points to combat inflation.",
        "Amazon's stock plummeted after missing earnings expectations and providing weak guidance.",
        "Microsoft's cloud revenue grew 30%, beating analyst forecasts.",
        "The merger was completed without any regulatory issues.",
        "Oil prices surged to a 10-year high amid supply concerns.",
        "The company filed for bankruptcy protection after defaulting on bond payments.",
        "Investors remained cautious ahead of tomorrow's inflation report."
    ]

    print("🔮 Testing FinBERT on Financial Statements")
    print("=" * 80)
    print()

    for i, statement in enumerate(financial_statements, 1):
        print(f"📊 Test {i}/{len(financial_statements)}")
        print("-" * 80)
        print(f"Statement: {statement}")
        print()

        sentiment, confidence, all_scores = classify_financial_sentiment(
            statement, model, tokenizer, device
        )

        print(f"Classify the sentiment (positive, negative, or neutral) of the following financial statement:")
        print(f"  ➜ Sentiment: {sentiment.upper()}")
        print(f"  ➜ Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print()
        print(f"All Scores:")
        print(f"  • Positive: {all_scores['positive']:.4f} ({all_scores['positive']*100:.2f}%)")
        print(f"  • Negative: {all_scores['negative']:.4f} ({all_scores['negative']*100:.2f}%)")
        print(f"  • Neutral:  {all_scores['neutral']:.4f} ({all_scores['neutral']*100:.2f}%)")
        print()
        print("=" * 80)
        print()

    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
