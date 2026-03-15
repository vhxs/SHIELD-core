FROM quay.io/pypa/manylinux_2_28_x86_64

RUN dnf update -y && dnf install -y curl wget

ENV PATH /opt/python/cp310-cp310/bin:${PATH}
ENV CMAKE_MODULE_PATH /opt/python/cp310-cp310/lib/cmake:/usr/local/lib64/cmake/

#
# install dependencies
#


RUN dnf groupinstall -y 'Development Tools' && \
    dnf install -y autoconf git

RUN pip install cmake && ln -s /opt/python/cp310-cp310/bin/cmake /usr/bin/cmake

# install openFHE
# Pull from the early-release version with bootstrapping
WORKDIR /openFHE

RUN git clone --recursive \
    https://github.com/openfheorg/openfhe-development.git/ && \
    cd openfhe-development && \
    mkdir openFHE-build && \
    cd openFHE-build && \
    cmake ../ -DBUILD_UNITTESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install


# string formatting
RUN git clone https://github.com/fmtlib/fmt.git && \
    cd fmt && \
    mkdir _build && cd _build && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. && \
    make -j$(nproc) && \
    make install

# Install numpy
RUN pip install "numpy<2.0.0"

# Install pybind11
RUN pip install "pybind11[global]>=2.11"

# openFHE is installed, now build the python packages
RUN mkdir openFHE-python
COPY . openFHE-python
RUN pip install -U uv
CMD set -ex; \
    cd /openFHE/openFHE-python && \
    uv build --wheel --out-dir /wheelhouse/tmp/; \
    cd /; \
    LD_LIBRARY_PATH=/opt/python/cp310-cp310/lib auditwheel repair -w /wheelhouse/ /wheelhouse/tmp/openfhe-*.whl; \
    chmod -R 777 ./wheelhouse
    
