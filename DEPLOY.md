# Deploying chunkops

This guide covers: local install, running tests, publishing to PyPI, and using in production.

---

## 1. Local development install

```bash
# Unzip the package
unzip chunkops.zip
cd chunkops

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Verify install
python -c "import chunkops; print(chunkops.__version__)"
# → 0.1.0
```

---

## 2. Run the full test suite

```bash
# All tests with verbose output
pytest

# With coverage report
pytest --cov=chunkops --cov-report=term-missing

# Run a specific test file
pytest tests/test_provenance.py -v

# Run a specific test
pytest tests/test_chunker.py::test_recursive_no_boundary_breaks -v
```

Expected output:

```
tests/test_tokenizer.py   ......        6 passed
tests/test_models.py      .......       7 passed
tests/test_chunker.py     ................  16 passed
tests/test_provenance.py  ............  12 passed
tests/test_benchmark.py   ..........   11 passed
tests/test_batch.py       ........      8 passed
─────────────────────────────────────────────────
60 passed in ~4s
```

---

## 3. Run the examples

```bash
# Basic usage walkthrough
python examples/basic_usage.py

# Full RAG pipeline simulation (no API key needed)
python examples/rag_pipeline.py
```

---

## 4. Try the CLI

```bash
# Create a test file
echo "The transformer architecture replaced recurrence with self-attention.
Each token attends to all others in the sequence.

BERT uses bidirectional encoders pre-trained with masked language modeling.
Fine-tuning BERT achieved state-of-the-art results in 2018." > test_doc.txt

# Chunk it
chunkops chunk test_doc.txt --strategy recursive

# Benchmark strategies
chunkops bench test_doc.txt --strategies fixed,recursive,structural
```

---

## 5. Publish to PyPI

### One-time setup

```bash
pip install build twine

# Create a PyPI account at https://pypi.org
# Then create an API token at https://pypi.org/manage/account/token/
```

### Build and upload

```bash
# Build source + wheel distributions
python -m build
# Creates: dist/chunkops-0.1.0.tar.gz
#          dist/chunkops-0.1.0-py3-none-any.whl

# Upload to TestPyPI first (safe test)
twine upload --repository testpypi dist/*
# Install from TestPyPI to verify:
pip install --index-url https://test.pypi.org/simple/ chunkops

# Upload to real PyPI
twine upload dist/*
```

### After publishing

```bash
pip install chunkops
pip install "chunkops[semantic]"
```

---

## 6. Production deployment tips

### Persist provenance across restarts

```python
# Use a file-backed store instead of in-memory
from chunkops import ProvenanceStore
store = ProvenanceStore("/var/data/chunkops_prov.db")
```

### Tune token limits for your embedding model

| Embedding model | Recommended max_tokens |
|----------------|------------------------|
| text-embedding-3-small | 512 |
| text-embedding-ada-002 | 512 |
| all-MiniLM-L6-v2 | 256 |
| BAAI/bge-large-en | 512 |
| nomic-embed-text | 2048 |

```python
chunker = Chunker(strategy="recursive", min_tokens=100, max_tokens=512)
```

### Large corpora — use BatchChunker with checkpoints

```python
from chunkops import BatchChunker

bc = BatchChunker(
    strategy="adaptive",
    workers=8,                         # set to CPU count for pure-Python strategies
    checkpoint="./ckpt/prod_run_001",  # never lose progress on crash
    min_tokens=100,
    max_tokens=512,
)
results = bc.run(document_iterator)
```

### Docker usage

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install ".[all]"
CMD ["python", "your_pipeline.py"]
```

---

## 7. Version bumping and release checklist

1. Update `version` in `pyproject.toml` and `chunkops/__init__.py`
2. Add entry to `CHANGELOG.md`
3. `pytest` — all green
4. `python -m build`
5. `twine check dist/*`
6. `twine upload dist/*`
7. `git tag v0.x.0 && git push --tags`
