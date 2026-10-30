# Stock SGLang image plus the target-feature capture patch that the Flash-Next DFlash2 draft
# repository ships (incoai/Qwen3.8-Flash-Next-DFlash2, patches/qwen38-dflash2-capture.patch).
# Without it, DFLASH on Qwen3.8-Flash-Next fails at the first forward pass.
ARG BASE
FROM ${BASE}
COPY patches/qwen38-dflash2-capture.patch /tmp/capture.patch
RUN cd /sgl-workspace/sglang && git apply /tmp/capture.patch && rm /tmp/capture.patch
