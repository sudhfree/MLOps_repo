FROM python:3.11.9 AS deploy
WORKDIR /app

#creating model and code folder to shift the files in container
RUN mkdir model code
COPY src/code/app.py /app/code
COPY src/code/requirements.txt /app/code
COPY src/code/train.py /app/code

#shifting model build on local system as we are not using remote hosted mlflow
COPY src/model/best_gradient_boosting_model.pkl /app/model

WORKDIR /app/code
#installing python dependencies
RUN pip install -r requirements.txt

#exposing 5000 port
EXPOSE 5000

CMD ["python", "app.py"]


