FROM ninai/microns-base
LABEL mantainer="Zhuokun Ding <zhuokund@bcm.edu>, Stelios Papadopoulos <spapadop@bcm.edu>"

# copy this project and install
COPY . /src/microns-nda
RUN pip install -e /src/microns-nda/python
