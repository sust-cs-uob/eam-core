FROM python:3.8-buster

RUN apt-get update
RUN apt-get install -y python3-matplotlib   libhdf5-serial-dev pandoc \
        texlive-latex-base texlive-fonts-recommended texlive-latex-extra graphviz

RUN mkdir -p /usr/src
WORKDIR /usr/src

#RUN docker pull markfletcher/graphviz
RUN pip install --no-deps git+https://github.com/hgrecco/pint.git@f356379c15c1cb5d211c795872ac9e9284d2358f#egg=pint
RUN pip install --no-deps git+https://github.com/crusaderky/pint.git@81cabd67722c657b3dae3c2d884b48f30313d9bd#egg=pint
# RUN pip install --no-deps git+https://github.com/hgrecco/pint-pandas.git#egg=pint-pandas

RUN pip3 install cython
RUN pip3 install numpy==1.18.2

# RUN pip install --no-deps git+https://github.com/sust-cs-uob/eam-data-tools.git@countries#egg=eam-data-tools
ADD eam-data-tools /usr/src/eam-data-tools
WORKDIR /usr/src/eam-data-tools
#COPY eam-data-tools/requirements.txt /usr/src/eam-data-tools/
#RUN pip3 install -r requirements.txt
RUN pip install .

WORKDIR /usr/src/eam-core
COPY eam-core/requirements.txt /usr/src/eam-core/
RUN pip3 install -r requirements.txt
COPY eam-core /usr/src/eam-core
RUN pip install .


#RUN python -m ngmodel.yaml_runner -a ci -l -c ci -f pdf tests/models/youtube.yml

EXPOSE 8080

#ENTRYPOINT ["/bin/bash"]
#CMD ["-m", "swagger_server"]
