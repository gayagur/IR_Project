import pandas as pd

data = [
    {
        "doc_id": 1,
        "title": "Computer science",
        "body": "Computer science is the study of computation and information."
    },
    {
        "doc_id": 2,
        "title": "Artificial intelligence",
        "body": "Artificial intelligence is intelligence demonstrated by machines."
    },
    {
        "doc_id": 3,
        "title": "Machine learning",
        "body": "Machine learning is a subset of artificial intelligence."
    },
    {
        "doc_id": 4,
        "title": "Information retrieval",
        "body": "Information retrieval deals with searching documents and text."
    }
]

df = pd.DataFrame(data)
df.to_parquet("sample.parquet", engine="pyarrow")

print("sample.parquet created")
