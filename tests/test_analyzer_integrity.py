# test_analyzer_integrity.py
import pandas as pd
from model_inference import calculate_metrics, analyze_sentiment

def test_metrics_accuracy():
    # Test case: Blended labels that could double-count
    data = [
        {"label": "Positive"},
        {"label": "Negative"},
        {"label": "Neutral (dominantly Positive)"},
        {"label": "Neutral"}
    ]
    df = pd.DataFrame(data)
    m = calculate_metrics(df)
    
    print(f"Metrics: {m}")
    total_pct = m["pos_pct"] + m["neg_pct"] + m["neu_pct"]
    print(f"Total Percentage: {total_pct}%")
    
    assert total_pct == 100.0, f"Total percentage should be 100%, but got {total_pct}%"
    assert m["pos_count"] == 1, "Positive count should be 1"
    assert m["neg_count"] == 1, "Negative count should be 1"
    assert m["neu_count"] == 2, "Neutral count should be 2 (including blended label)"
    print("Metrics test passed!")

if __name__ == "__main__":
    try:
        test_metrics_accuracy()
    except Exception as e:
        print(f"Test failed: {e}")
