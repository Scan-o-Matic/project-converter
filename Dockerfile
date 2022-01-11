FROM python:2.7

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY data/ /tmp/data/
COPY scripts/ /tmp/scripts/
COPY scanomatic/ /tmp/scanomatic/
COPY setup.py /tmp/setup.py
COPY setup_tools.py /tmp/setup_tools.py

RUN cd /tmp && python setup.py install --default
COPY converter.py /app/
WORKDIR /app
CMD ./converter.py
