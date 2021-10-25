ARG BASE_IMAGE=at-docker:5000/zhuokund/pytorch

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=spapa013

# GitHub username and GitHub Personal Access Token must be specified
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials

RUN git clone https://github.com/${DEV_SOURCE}/utils.git

# Building the second stage
FROM ${BASE_IMAGE}
LABEL mantainer="Zhuokun Ding <zhuokund@bcm.edu>"
# copy everything found in /data over
# and then install them
COPY --from=base /src /src

RUN pip install /src/utils
RUN pip install pycircstat nose tables

# copy this project and install
COPY . /src/microns-nda
RUN pip install -e /src/microns-nda/python
