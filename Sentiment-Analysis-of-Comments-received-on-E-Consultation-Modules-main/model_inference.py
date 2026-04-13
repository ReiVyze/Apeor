import torch
import os
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "model")
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

class SentimentAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.vader = SentimentIntensityAnalyzer()
        return cls._instance

    def _get_roberta_pipeline(self):
        if "roberta" not in self.models:
            self.models["roberta"] = pipeline(
                "sentiment-analysis",
                model=ROBERTA_MODEL,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        return self.models["roberta"]

    def _get_local_distilbert_pipeline(self):
        if "distilbert" not in self.models:
            self.models["distilbert"] = pipeline(
                "sentiment-analysis",
                model=MODEL_PATH_LOCAL,
                tokenizer=MODEL_PATH_LOCAL,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        return self.models["distilbert"]

    def analyze(self, text: str, model_type: str = "RoBERTa (HF)", aspect: str = "General"):
        if model_type == "RoBERTa (HF)":
            res = self._analyze_roberta(text)
        elif model_type == "Local DistilBert":
            res = self._analyze_distilbert(text)
        else:
            res = self._analyze_vader(text)
        
        res["aspect"] = aspect
        res["polarity"] = self._get_polarity(res["label"], res["score"])
        return res

    def _get_polarity(self, label: str, score: float):
        """Maps label and confidence to a -1.0 to 1.0 polarity scale."""
        clean_label = "Neutral"
        if "Positive" in label: clean_label = "Positive"
        elif "Negative" in label: clean_label = "Negative"

        if clean_label == "Positive": return score
        if clean_label == "Negative": return -score
        return 0.0

    def _analyze_vader(self, text: str):
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05: label = "Positive"
        elif compound <= -0.05: label = "Negative"
        else: label = "Neutral"
        return {"comment": text, "label": label, "score": abs(compound)}

    def _analyze_roberta(self, text: str):
        pipe = self._get_roberta_pipeline()
        # Explicit truncation to 512 tokens
        result = pipe(text, truncation=True, max_length=512)[0]
        # Twitter-roberta-base labels: LABEL_0 -> Negative, LABEL_1 -> Neutral, LABEL_2 -> Positive
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        top_res = max(result, key=lambda x: x['score'])
        label = label_map.get(top_res['label'], top_res['label'])
        
        neutral_score = next((x['score'] for x in result if x['label'] == "LABEL_1"), 0.0)
        if neutral_score > 0.4 and label != "Neutral":
            label = f"Neutral (dominantly {label})"
            score = neutral_score
        else:
            score = top_res['score']
        
        return {"comment": text, "label": label, "score": score}

    def _analyze_distilbert(self, text: str):
        pipe = self._get_local_distilbert_pipeline()
        result = pipe(text, truncation=True, max_length=512)[0]
        top_res = max(result, key=lambda x: x['score'])
        return {"comment": text, "label": top_res['label'], "score": top_res['score']}

    def analyze_batch(self, texts: list, model_type: str = "RoBERTa (HF)", aspect: str = "General"):
        """True batch inference using the transformers pipeline."""
        if not texts: return []
        if model_type == "Vader (Lexicon)":
            return [self.analyze(t, model_type, aspect) for t in texts]

        pipe = self._get_roberta_pipeline() if model_type == "RoBERTa (HF)" else self._get_local_distilbert_pipeline()
        
        # Ensure texts is a list
        if isinstance(texts, str): texts = [texts]
        
        results = pipe(texts, batch_size=16, truncation=True, max_length=512)
        
        final_results = []
        for i, res in enumerate(results):
            # The pipeline usually returns a list of results (one per text).
            # If return_all_scores=True, each result is a list of dictionaries.
            # If return_all_scores=False, each result is a single dictionary.
            
            if isinstance(res, list):
                # We have all scores, find the top one
                top_res = max(res, key=lambda x: x['score'])
            else:
                # We already have the top score as a dict
                top_res = res
                
            if model_type == "RoBERTa (HF)":
                label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
                label = label_map.get(top_res['label'], top_res['label'])
                score = top_res['score']
            else:
                label = top_res['label']
                score = top_res['score']
            
            final_res = {"comment": texts[i], "label": label, "score": score, "aspect": aspect}
            final_res["polarity"] = self._get_polarity(label, score)
            final_results.append(final_res)
        return final_results



    def calculate_metrics(self, df):
        if df.empty: return {"total": 0, "pos_pct": 0, "neg_pct": 0, "neu_pct": 0, "nps": 0, "csat": 0}
        total = len(df)
        
        # Sentiment breakdown
        pos_count = len(df[df["label"] == "Positive"])
        neg_count = len(df[df["label"] == "Negative"])
        neu_count = total - pos_count - neg_count
        
        # NPS Calculation (if 'score' or 'rating' exists)
        # Assuming mock logic for now where high sentiment scores correlate with NPS status
        nps_score = 0
        if 'nps_rating' in df.columns:
            promoters = len(df[df['nps_rating'] >= 9])
            detractors = len(df[df['nps_rating'] <= 6])
            nps_score = round(((promoters - detractors) / total) * 100)
        
        # CSAT Calculation (if 'csat_rating' exists)
        csat_avg = 0
        if 'csat_rating' in df.columns:
            csat_avg = round(df['csat_rating'].mean(), 1)
            
        return {
            "total": total, "pos_count": pos_count, "neg_count": neg_count, "neu_count": neu_count,
            "pos_pct": round((pos_count/total)*100, 1), "neg_pct": round((neg_count/total)*100, 1), 
            "neu_pct": round((neu_count/total)*100, 1),
            "nps": nps_score, "csat": csat_avg
        }

    def get_actionable_category(self, text: str):
        """Categorize feedback into actionable business buckets."""
        text = text.lower()
        if any(w in text for w in ['price', 'cost', 'expensive', 'cheap', 'funding', 'tax', 'economics']):
            return "Price/Economics"
        if any(w in text for w in ['service', 'support', 'help', 'representative', 'call', 'chat', 'portal']):
            return "Service/Support"
        if any(w in text for w in ['policy', 'clause', 'initiative', 'product', 'feature', 'system', 'tech', 'healthcare', 'education']):
            return "Product/Policy"
        return "General"

# Singleton
analyzer = SentimentAnalyzer()

def analyze_sentiment(comment: str, model_type: str = "RoBERTa (HF)", aspect: str = "General"):
    return analyzer.analyze(comment, model_type, aspect)

def analyze_batch(comments: list, model_type: str = "RoBERTa (HF)", aspect: str = "General"):
    return analyzer.analyze_batch(comments, model_type, aspect)

def calculate_metrics(df):
    return analyzer.calculate_metrics(df)

def get_actionable_category(text: str):
    return analyzer.get_actionable_category(text)
