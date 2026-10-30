# GSM8K client: sgl-eval's prompt and grader (math_verify) on the dataset's parquet file.
# sgl-eval is installed without dependencies: only its vendored NeMo-Skills grader is used,
# and its other requirements (datasets, pandas, pillow, ...) are not needed for that.
FROM python:3.12-slim
RUN pip install --no-cache-dir --no-deps sgl-eval==0.1.3 \
 && pip install --no-cache-dir "math-verify[antlr4_9_3]==0.9.0" latex2sympy2-extended==1.11.0 \
        pyyaml==6.0.3 pyarrow==26.0.0
WORKDIR /out
