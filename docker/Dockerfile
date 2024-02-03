FROM python:3.10

EXPOSE 8989
 
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src/app /code/app
COPY ./models /models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8989"]