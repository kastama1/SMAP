FROM python:3.9

WORKDIR /smart-recycling

COPY requirements.txt .
COPY ./src ./src
COPY ./model ./model
COPY ./test ./test

RUN pip install -r requirements.txt

CMD ["python", "./src/main.py"]