#!/bin/bash
cd `dirname $0`

# Quiet Hugging Face / tqdm progress bars once the module process is running
# (weight load still happens in-process; download is done at setup below).
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

if [ -f .installed ]
  then
    source viam-env/bin/activate
  else
    python3 -m pip install --user virtualenv --break-system-packages
    python3 -m venv viam-env
    source viam-env/bin/activate
    pip3 install --upgrade -r requirements.txt
    if [ $? -eq 0 ]
      then
        # Download default weights now so configure-time logs stay clean.
        # Override with QWEN3_VL_MODEL if you use a non-default checkpoint.
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
