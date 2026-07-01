# GeoScout backend image — the FastMCP server only (geocode, find_pois,
# compute_distance_matrix, score_sites, plus GET /health). The agent and
# Streamlit UI run as separate processes/deployments and are not part of
# this image.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend/ ./backend/

# Render (and most PaaS hosts) inject PORT at runtime; default to 8000 for
# local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "backend.server"]
