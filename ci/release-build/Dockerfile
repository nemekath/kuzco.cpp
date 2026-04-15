FROM rocm/dev-ubuntu-22.04:7.1-complete

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /build /out && chmod 777 /build /out

WORKDIR /build
