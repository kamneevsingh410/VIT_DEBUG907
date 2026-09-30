FROM python:3.14-slim

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt && python -m pip check

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('Alibaba-NLP/gte-modernbert-base', device='cpu', revision='e7f32e3c00f91d699e8c43b53106206bcc72bb22'); \
d = m.get_sentence_embedding_dimension(); \
assert d == 768, d; \
print('baked in: gte-modernbert-base,', d, 'dims')"

RUN python -c "\
from datasets import load_dataset; \
R = 'f22508f96b7a36c2415181ed8bb76f76e04ae2d5'; \
Q = 'a4fb4d92996bcfe1e0b0af9e97ea4b70b80ec9d5'; \
c = load_dataset('CoIR-Retrieval/apps', 'corpus', split='corpus', revision=R); \
q = load_dataset('CoIR-Retrieval/apps', 'queries', split='queries', revision=R); \
r = load_dataset('CoIR-Retrieval/apps-qrels', split='test', revision=Q); \
print('baked in: AppsRetrieval rev', R[:8], len(c), 'docs,', len(q), 'queries,', len(r), 'qrels')"

COPY . /app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN python -c "\
from retrieval.embed import MAX_SEQ_LENGTH, MODEL_NAME; \
assert MODEL_NAME == 'Alibaba-NLP/gte-modernbert-base', MODEL_NAME; \
assert MAX_SEQ_LENGTH == 1024, MAX_SEQ_LENGTH; \
print('shipped config: ', MODEL_NAME, 'seq', MAX_SEQ_LENGTH)" \
 && python -m pytest -q \
 && python cli.py selftest

RUN useradd --create-home --uid 1000 debug907 \
 && mkdir -p /app/out \
 && chown -R debug907 /app/out /opt/hf
USER debug907

ENTRYPOINT ["python", "cli.py"]
CMD []
