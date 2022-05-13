FROM at-docker.ad.bcm.edu:5000/microns-base
LABEL mantainer="Zhuokun Ding <zhuokund@bcm.edu>, Stelios Papadopoulos <spapadop@bcm.edu>"

# copy this project and install
COPY . /src/microns-nda
RUN pip3 install -e /src/microns-nda/python/microns-nda
RUN pip3 install -e /src/microns-nda/python/microns-nda-api
