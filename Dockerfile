# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

COPY . /app

# Install system dependencies and link python3 to python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch first
RUN pip install torch==2.3.0
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
RUN pip install torch_geometric
# Install remaining Python packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt --timeout=1000
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# Verify installations
RUN python -c "import numpy; print('NumPy version:', numpy.__version__)"
RUN python -c "import torch; print('Torch version:', torch.__version__)"
RUN python -c "import torch_geometric; print('PyTorch Geometric installed')"

# Define the command to run your application
CMD ["bash"]