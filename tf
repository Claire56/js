import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def search_by_tf(query: str, documents: List[str], n: int) -> List[Tuple[int, float]]:
    """
    Search documents using TF (Term Frequency) and return top n results.
    
    Args:
        query: Search query string
        documents: List of document strings
        n: Number of top results to return
        
    Returns:
        List of tuples (document_index, tf_score) sorted by score descending
    """
    if not query or not documents or n <= 0:
        return []
    
    # Preprocess query and documents
    query_terms = preprocess_text(query)
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # Calculate TF scores
    tf_scores = calculate_tf_scores(query_terms, processed_docs)
    
    # Sort by score and return top n
    sorted_results = sorted(tf_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:n]

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by tokenizing and normalizing.
    
    Args:
        text: Input text string
        
    Returns:
        List of normalized tokens
    """
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into tokens and remove empty strings
    tokens = [token for token in text.split() if token]
    
    return tokens

def calculate_tf_scores(query_terms: List[str], processed_docs: List[List[str]]) -> List[Tuple[int, float]]:
    """
    Calculate TF (Term Frequency) scores for documents given query terms.
    
    Args:
        query_terms: List of query tokens
        processed_docs: List of processed document token lists
        
    Returns:
        List of (document_index, tf_score) tuples
    """
    if not query_terms or not processed_docs:
        return []
    
    scores = []
    
    for doc_idx, doc_terms in enumerate(processed_docs):
        if not doc_terms:  # Skip empty documents
            continue
            
        # Calculate term frequency (TF) for this document
        term_freq = Counter(doc_terms)
        doc_length = len(doc_terms)
        
        # Calculate TF score for this document
        doc_score = 0.0
        
        for term in query_terms:
            if term in term_freq:
                # TF: term frequency in document (normalized by document length)
                tf = term_freq[term] / doc_length
                doc_score += tf
        
        scores.append((doc_idx, doc_score))
    
    return scores

def get_document_tf_stats(documents: List[str]) -> Dict[int, Dict[str, float]]:
    """
    Get TF statistics for all documents.
    
    Args:
        documents: List of document strings
        
    Returns:
        Dictionary mapping document index to term frequency dictionary
    """
    processed_docs = [preprocess_text(doc) for doc in documents]
    tf_stats = {}
    
    for doc_idx, doc_terms in enumerate(processed_docs):
        if not doc_terms:
            tf_stats[doc_idx] = {}
            continue
            
        term_freq = Counter(doc_terms)
        doc_length = len(doc_terms)
        
        # Calculate normalized TF for each term
        tf_dict = {term: freq / doc_length for term, freq in term_freq.items()}
        tf_stats[doc_idx] = tf_dict
    
    return tf_stats

# ============================================================================
# TEST CASES
# ============================================================================

def run_test_cases():
    """Run test cases to verify the TF search function."""
    print("Testing TF-Based Document Search\n")
    
    # Test documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy fox sleeps while the quick brown dog watches",
        "A document about machine learning and artificial intelligence",
        "Machine learning algorithms process data to find patterns",
        "Artificial intelligence systems can learn and adapt",
        "Data science involves statistics, machine learning, and programming",
        "Python is a popular programming language for data science"
    ]
    
