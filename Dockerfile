FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
LABEL mantainer="Zhuokun Ding <zhuokund@bcm.edu>, Stelios Papadopoulos <spapadop@bcm.edu>"
# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Install essential Ubuntu packages
# and upgrade pip
RUN apt-get update &&\
    apt-get install -y software-properties-common \
                       build-essential \
                       git \
                       wget \
                       vim \
                       curl \
                       zip \
                       zlib1g-dev \
                       unzip \
                       pkg-config \
                       libblas-dev \
                       liblapack-dev \
                       python3-tk \
                       python3-wheel \
                       graphviz \
                       libhdf5-dev \
                       python3.8 \
                       python3.8-dev \
                       python3.8-distutils \
                       swig &&\
    apt-get clean &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Node.js for rebuilding jupyter lab
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
RUN apt-get install -y nodejs

WORKDIR /src

# Install essential Python packages
RUN python3 -m pip --no-cache-dir install \
         pytest \
         pytest-cov \
         numpy \
         matplotlib \
         scipy \
         pandas \
         jupyterlab \
         ipywidgets \
         ipympl \
         scikit-learn \
         scikit-image \
         seaborn \
         graphviz \
         h5py

RUN python3 -m pip --no-cache-dir install git+https://github.com/spapa013/datajoint-python.git

# Add profiling library support
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Export port for Jupyter Notebook
EXPOSE 8888

# copy this project and install
COPY . /src/microns-nda
RUN pip install -e /src/microns-nda/python
