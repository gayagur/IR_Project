# text_processing.py
import re
from typing import List

# ----------------------------
# Tokenizer (from Assignment 3 – GCP part)
# ----------------------------

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# ----------------------------
# Stopwords (from Assignment 3)
# ----------------------------
STOPWORDS = {
    'the','of','and','to','in','a','is','it','you','that','he','was','for','on',
    'are','with','as','i','his','they','be','at','one','have','this','from','or',
    'had','by','hot','word','but','what','some','we','can','out','other','were',
    'all','there','when','up','use','your','how','said','an','each','she'
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text according to Assignment 3 (GCP tokenizer).
    - regex based
    - lowercase
    - remove stopwords
    - NO stemming
    """
    if not text:
        return []

    tokens = [t.group().lower() for t in RE_WORD.finditer(text)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
