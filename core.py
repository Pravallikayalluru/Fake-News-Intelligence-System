"""Shared logic: cleaning, shortcut masking, style features, and red-flag rules."""
import re
import numpy as np
from textblob import TextBlob

SENSATIONAL = [
    "breaking", "shocking", "secret", "revealed", "viral", "miracle",
    "you won't believe", "exposed", "bombshell", "outrage", "destroyed",
    "hoax", "conspiracy", "cover-up", "they don't want you",
]
ATTRIBUTION = [
    "said", "according to", "told", "reported", "spokesman",
    "spokeswoman", "statement", "officials",
]

ARTIFACT_PATTERNS = [
    r"\b[A-Z][A-Za-z ,\.]{2,40}\((?:reuters|ap|afp)\)\s*[-—–]?",
    r"\b(?:reuters|associated press|cnn|breitbart|infowars|new york times|"
    r"washington post|fox news|daily mail|politico|nytimes)\b",
    r"\b(?:share|tweet|print|email|advertisement|subscribe|comments?|"
    r"follow us|read more|click here)\b",
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    r"\d+",
]

def mask_artifacts(text: str) -> str:
    for pattern in ARTIFACT_PATTERNS:
        text = re.sub(pattern, " ", str(text), flags=re.I)
    return text

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def prep(text: str) -> str:
    """Preprocessing used by the selected artifact-masked model."""
    return clean_text(mask_artifacts(text))

FEATURES = [
    "log_words", "caps_ratio", "exclaim_rate", "question_rate",
    "avg_word_len", "sensational_rate", "attribution_rate",
    "quote_rate", "subjectivity", "polarity_abs",
]

LABELS = {
    "caps_ratio": "ALL-CAPS words",
    "exclaim_rate": "Exclamation marks",
    "question_rate": "Question marks",
    "sensational_rate": "Sensational wording",
    "subjectivity": "Subjective / opinionated tone",
    "polarity_abs": "Emotional intensity",
    "attribution_rate": "Named sources / attribution",
    "quote_rate": "Direct quotes",
}
RISK_IF_HIGH = [
    "caps_ratio", "exclaim_rate", "question_rate",
    "sensational_rate", "subjectivity", "polarity_abs",
]
RISK_IF_LOW = ["attribution_rate", "quote_rate"]

def style_features(text: str) -> dict:
    t = str(text)[:5000]
    words = re.findall(r"[A-Za-z']+", t)
    n = max(len(words), 1)
    low = t.lower()
    sentiment = TextBlob(t).sentiment
    all_words = re.findall(r"[A-Za-z']+", str(text))
    return {
        "log_words": float(np.log1p(len(all_words))),
        "caps_ratio": sum(1 for w in words if len(w) > 1 and w.isupper()) / n,
        "exclaim_rate": t.count("!") / n * 100,
        "question_rate": t.count("?") / n * 100,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "sensational_rate": sum(low.count(w) for w in SENSATIONAL) / n * 100,
        "attribution_rate": sum(low.count(a) for a in ATTRIBUTION) / n * 100,
        "quote_rate": (t.count('"') + t.count("“")) / 2 / n * 100,
        "subjectivity": float(sentiment.subjectivity),
        "polarity_abs": abs(float(sentiment.polarity)),
    }

def style_vector(text: str) -> np.ndarray:
    features = style_features(text)
    return np.array([features[key] for key in FEATURES], dtype=float)

def build_baseline(style_matrix: np.ndarray, is_real: np.ndarray) -> dict:
    """Calculate REAL-class training-set reference statistics."""
    real = style_matrix[is_real]
    if len(real) == 0:
        raise ValueError("The training split contains no REAL examples.")
    return {
        key: {
            "median": float(np.median(real[:, i])),
            "p10": float(np.percentile(real[:, i], 10)),
            "p90": float(np.percentile(real[:, i], 90)),
        }
        for i, key in enumerate(FEATURES)
    }

def red_flags(feats: dict, baseline: dict) -> list:
    flags = []
    for key in RISK_IF_HIGH:
        ref = baseline[key]
        if feats[key] > ref["p90"] and feats[key] > 0:
            if ref["median"] < 0.05:
                message = "rarely appears in real news"
            else:
                message = f"{min(feats[key] / ref['median'], 99):.1f}x higher than typical real news"
            flags.append((LABELS[key], message))
    for key in RISK_IF_LOW:
        if feats[key] < baseline[key]["p10"]:
            flags.append((LABELS[key], "lower than almost all real-news articles"))
    return flags
