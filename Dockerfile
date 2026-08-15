# Souprise demo container: web GUI + CPU inference, everything local.
# Build:  docker compose up --build
# Then open http://localhost:8501
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY souprise ./souprise
COPY examples ./examples
COPY benchmarks ./benchmarks
COPY .streamlit ./.streamlit

# Core + GUI + importers, then CPU-only PyTorch (much smaller than the
# CUDA build) and transformers for local generation.
RUN pip install --no-cache-dir -e ".[gui,excel]" \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers

EXPOSE 8501

# The GUI starts in search-only mode; enabling answers downloads a small
# model once into the mounted cache volume.
CMD ["python", "-m", "streamlit", "run", "souprise/gui/app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--browser.gatherUsageStats", "false"]
