#!/bin/bash
cd `dirname $0`

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

install_requirements() {
  # Build llama-cpp-python with the best available accelerator.
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "Installing deps with llama.cpp Metal support..."
    CMAKE_ARGS="-DGGML_METAL=on" pip3 install --upgrade -r requirements.txt
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing deps with llama.cpp CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip3 install --upgrade -r requirements.txt
  else
    echo "Installing deps with llama.cpp CPU support..."
    pip3 install --upgrade -r requirements.txt
  fi
}

if [ -f .installed ]
  then
    source viam-env/bin/activate
  else
    python3 -m pip install --user virtualenv --break-system-packages
    python3 -m venv viam-env
    source viam-env/bin/activate
    install_requirements
    if [ $? -eq 0 ]
      then
        # Download default GGUF + mmproj so configure-time logs stay clean.
        python3 prefetch_model.py
        if [ $? -eq 0 ]
          then
            touch .installed
        fi
    fi
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec python3 -m src $@
