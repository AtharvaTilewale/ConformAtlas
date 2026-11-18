# Base image. The ICR workflow resolves this tag to an exact digest at build
# time, passes it in as BASE_IMAGE=python:3.11-slim@sha256:..., and records it
# in the image labels, so a build can be repeated on the identical base.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Release of ConformAtlas installed from PyPI (override with --build-arg).
ARG CONFORMATLAS_VERSION=1.0.3
ARG BASE_IMAGE

LABEL org.opencontainers.image.title="ConformAtlas" \
      org.opencontainers.image.description="Mapping conformational landscapes from molecular simulations" \
      org.opencontainers.image.version="${CONFORMATLAS_VERSION}" \
      org.opencontainers.image.source="https://github.com/AtharvaTilewale/ConformAtlas" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Atharva Tilewale, Dhaval Patel" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Debian security updates, plus tini so the container stops promptly.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

# The numerical stack that determines results is pinned to the versions
# ConformAtlas was validated with (taken from requirements-lock.txt). Other
# dependencies (pillow, filelock, ...) float to current releases so security
# fixes are picked up. pip, setuptools and wheel are removed afterwards: they
# are only needed to install, and their bundled copies carry fixable CVEs.
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN grep -iE '^(numpy|scipy|pandas|MDAnalysis|scikit-image|matplotlib)==' \
        /tmp/requirements-lock.txt > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
        "conformatlas[interactive,hdf5,progress]==${CONFORMATLAS_VERSION}" \
 && conformatlas --version \
 && python -m pip uninstall -y pip setuptools wheel \
 && rm -rf /tmp/requirements-lock.txt /tmp/constraints.txt /usr/local/lib/python3*/ensurepip/_bundled \
 && useradd --no-log-init --create-home --uid 100 --gid users conformatlas \
 && mkdir -p /data && chown conformatlas:users /data

# ICE Cloud mounts each user's drive at /drive/<username>, owned by 100:100
# (group "users") with mode 755, and replaces the start command with its own
# web terminal, so the image's default user is who you are in the terminal.
# Running as 100:100 lets ConformAtlas write results to the persistent drive.
USER 100:100
WORKDIR /data

ENTRYPOINT ["/usr/bin/tini", "--"]

# Keeps an ICE Cloud instance running so users can open its terminal and run
# `conformatlas analyze ...`. With plain Docker:
#   docker run --rm -v "$PWD:/data" <image> conformatlas analyze -s top.pdb -f traj.xtc -o out
CMD ["sleep", "infinity"]
