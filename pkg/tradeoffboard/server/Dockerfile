FROM python:3.11-alpine

WORKDIR /tradeoffboard
COPY ./pkg/tradeoffboard/server .
RUN pip3 install -r requirements.txt


ENTRYPOINT ["python3", "secure.py"]
CMD ["--port", "50051", "--log", "debug"]
