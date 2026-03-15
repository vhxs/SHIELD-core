# OpenFHE Python bindings

Python bindings for OpenFHE's implementation of the CKKS encryption scheme, including arithmetic operations, bootstrapping, and neural network operations. This README has instructions for building a Python wheel file, which can be imported as a Python module for use in Python codebases. These bindings are a dependency for [code developed](https://github.com/JHUAPL/SHIELD) that accompanies the manuscript: [High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts](https://arxiv.org/abs/2306.09189).

## Requirements

- **Docker** — the wheel is built inside a container; all dependencies (OpenFHE, fmt, pybind11, numpy) are downloaded and installed during the image build.
- **x86-64 host** — the base image is `manylinux_2_34_x86_64`. The build is untested on ARM or other architectures.
- The resulting wheel requires **Python 3.12**, **numpy ≥ 2.0.0**, and a Linux system with **glibc ≥ 2.34** (Ubuntu 22.04+).

## Build Instructions

Build the Docker image (this installs OpenFHE and all other dependencies — expect ~10 minutes on first build):

```bash
docker build . --tag openfhe-python-build
```

Run the container to produce the wheel. The wheel is written to `./wheelhouse/` on the host:

```bash
docker run -v ${PWD}/wheelhouse:/wheelhouse openfhe-python-build
```

Install the wheel in a Python environment:

```bash
pip install wheelhouse/openfhe-*.whl
```

Once installed, the bindings can be imported with `import pyOpenFHE`. See the [SHIELD repository](https://github.com/JHUAPL/SHIELD) for usage examples.

## Running Tests

After building the image and wheel, run the test suite inside the container:

```bash
docker run --rm -v ${PWD}/wheelhouse:/wheelhouse --entrypoint /bin/bash openfhe-python-build -c "
  pip install /wheelhouse/openfhe-*.whl -q && pip install pytest -q &&
  cd /openFHE/openFHE-python && python -m pytest -m slow"
```

## Citation and Acknowledgements

Please cite this work if using it on other projects. In addition to the authors on the supporting manuscript (Vivian Maloney, Freddy Obrecht, Vikram Saraph, Prathibha Rama, and Kate Tallaksen), Aaron Pendergrass and Charlie Schneider also made significant contributions to this work.
