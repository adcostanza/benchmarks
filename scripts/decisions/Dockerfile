# Typed Decisions client: stdlib HTTP, pyarrow for the parquet files, Pillow to convert non-RGB
# images (COCO has grayscale JPEGs; SGLang rejects 1-channel input).
FROM python:3.12-slim
RUN pip install --no-cache-dir pyarrow==26.0.0 pillow==10.2.0
WORKDIR /out
