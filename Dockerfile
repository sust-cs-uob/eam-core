FROM python:3.8-buster


RUN apt-get update
RUN apt-get install -y python3-matplotlib cython3  libhdf5-serial-dev pandoc \
        texlive-latex-base texlive-fonts-recommended texlive-latex-extra graphviz

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app


COPY requirements.txt /usr/src/app/
RUN pip3 install -r requirements.txt
#RUN docker pull markfletcher/graphviz

RUN pip install --no-deps git+https://github.com/hgrecco/pint-pandas.git#egg=pint-pandas

COPY . /usr/src/app
RUN pip install .
#RUN python -mpip install -U matplotlib

#RUN python -m ngmodel.yaml_runner -a ci -l -c ci -f pdf tests/models/youtube.yml

EXPOSE 8080

#ENTRYPOINT ["/bin/bash"]
#CMD ["-m", "swagger_server"]