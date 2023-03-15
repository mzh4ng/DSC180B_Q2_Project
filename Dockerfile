# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

RUN pip install numpy==1.22

RUN pip install matplotlib==3.7.1

RUN pip install scikit-learn==1.2.1

RUN pip install pandas==1.5.3

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]