FROM python:3.11-slim

# Release of ConformAtlas installed from PyPI (override with --build-arg).
ARG CONFORMATLAS_VERSION=1.0.2

LABEL org.opencontainers.image.title="ConformAtlas" \
      org.opencontainers.image.description="Mapping conformational landscapes from molecular simulations" \
      org.opencontainers.image.version="${CONFORMATLAS_VERSION}" \
      org.opencontainers.image.source="https://github.com/AtharvaTilewale/ConformAtlas" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Atharva Tilewale, Dhaval Patel"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN pip install --no-cache-dir "conformatlas[interactive,hdf5,progress]==${CONFORMATLAS_VERSION}" \
 && conformatlas --version \
 && useradd --create-home --uid 1000 conformatlas \
 && mkdir -p /data && chown conformatlas:conformatlas /data

USER 1000
WORKDIR /data

# Keeps an ICE Cloud instance running so users can open its terminal and run
# `conformatlas analyze ...`. With plain Docker:
#   docker run --rm -v "$PWD:/data" <image> conformatlas analyze -s top.pdb -f traj.xtc -o out
CMD ["sleep", "infinity"]
