"""Shared pytest fixtures."""
import pytest

SHORT_DOC = (
    "The transformer architecture replaced recurrence with self-attention. "
    "Each token attends to all others in the sequence."
)

MEDIUM_DOC = """
The transformer architecture, introduced in 'Attention is All You Need' (2017),
replaced recurrence with self-attention. Each token attends to all others in the
sequence, enabling parallel computation and capturing long-range dependencies.

BERT uses bidirectional encoders pre-trained with masked language modeling.
It learns deep context from both left and right of each token. Fine-tuning BERT
on downstream tasks like classification achieved state-of-the-art results in 2018.

GPT models use decoder-only transformers trained autoregressively. They predict
the next token given all previous tokens. Scaling GPT to billions of parameters
led to emergent capabilities like few-shot and zero-shot generalization.

Vector databases store high-dimensional embeddings produced by models like these.
Approximate nearest-neighbor search retrieves the k most similar vectors to a
query in milliseconds. Pinecone, Qdrant, and pgvector are common choices.
""".strip()

MARKDOWN_DOC = """
# Introduction to RAG

Retrieval-Augmented Generation combines retrieval systems with language models.
It grounds responses in factual source documents.

## How it works

The system first retrieves relevant document chunks using vector similarity.
Then it passes those chunks as context to the language model.
The model generates a response grounded in the retrieved content.

## Chunking strategies

Choosing the right chunking strategy is critical for RAG performance.
Poor chunks lead to poor retrieval, which leads to poor answers.
""".strip()

EMPTY_DOC = ""
WHITESPACE_DOC = "   \n\n   "


@pytest.fixture
def short_doc():
    return SHORT_DOC

@pytest.fixture
def medium_doc():
    return MEDIUM_DOC

@pytest.fixture
def markdown_doc():
    return MARKDOWN_DOC
