FROM python:3.12-slim
# bfcl-eval 2026.3.23 imports soundfile without declaring it.
RUN pip install --no-cache-dir bfcl-eval==2026.3.23 soundfile
WORKDIR /out
