cd ~/Documents/CudaTags

mkdir -p build
cd build
cmake ..
make -j 5

cd ..

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# for python 3.10 on aarch64
cp build/cuda_tags.cpython-310-aarch64-linux-gnu.so .venv/lib/python3.10/site-packages/

# Make sure pybind11-stubgen is found via the current environment's bin
PYTHON_BIN=$(which python)
PYBIND11_STUBGEN_BIN="$(dirname "$PYTHON_BIN")/pybind11-stubgen"
"$PYBIND11_STUBGEN_BIN" cuda_tags --output-dir .venv/lib/python3.10/site-packages/