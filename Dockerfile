FROM at-docker:5000/microns-base:cuda11.8.0-python3.8
LABEL mantainer="Zhuokun Ding <zhuokund@bcm.edu>, Stelios Papadopoulos <spapadop@bcm.edu>"

# copy this project and install
COPY . /src/microns-nda
RUN pip3 install -e /src/microns-nda/python/microns-nda
RUN pip3 install -e /src/microns-nda/python/microns-nda-api
