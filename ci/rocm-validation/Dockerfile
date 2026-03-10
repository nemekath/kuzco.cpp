# ROCm 7.2 test container for T-MAC / kuzco.cpp
# Purpose: Isolated build+bench environment — host ROCm 7.1.1 stays untouched.
#
# Usage:
#   docker build -f Dockerfile.rocm72-test -t kuzco-rocm72-test .
#   ./rocm72-test.sh          # interactive shell with GPU access
#   ./rocm72-test.sh build    # build inside container
#   ./rocm72-test.sh bench    # run quick benchmark
#
FROM rocm/dev-ubuntu-24.04:7.2-complete

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Reuse existing ubuntu user (UID 1000) from base image, add GPU groups
RUN usermod -aG video,render ubuntu

# Build directory lives inside container (not on host volume)
RUN mkdir -p /build && chown ubuntu:ubuntu /build

USER ubuntu
WORKDIR /build

# Source and models are bind-mounted at runtime (read-only)
# /src  -> llama-tmac-master (ro)
# /models -> gguf models (ro)

COPY --chown=ubuntu:ubuntu rocm72-entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["shell"]
