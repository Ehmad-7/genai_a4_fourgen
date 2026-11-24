"""Part D: split abstracts into chunks of at most 150 words with a 30-word overlap."""
MAX_WORDS, OVERLAP = 150, 30


def chunk_words(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Consecutive chunks share `overlap` words; the last chunk ends at the last word of the abstract."""
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    return chunks


def build_chunks(corpus):
    """corpus {abstract_id: text} -> (chunk_texts, chunk_doc_ids), in corpus order.
    Every chunk remembers the id of its abstract (same position in chunk_doc_ids)."""
    texts, ids = [], []
    for doc_id, text in corpus.items():
        for c in chunk_words(text):
            texts.append(c)
            ids.append(doc_id)
    return texts, ids
