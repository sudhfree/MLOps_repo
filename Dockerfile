# FROM python:3.11.9 AS build
# WORKDIR /app
# RUN mkdir data
# COPY ./data/housing.csv /app/data
# COPY ./app.py ./requirements.txt ./train.py /app/
# RUN ls -al
# RUN ls -al data
# RUN pip install -r requirements.txt && mkdir model
# RUN python train.py

FROM python:3.11.9 AS deploy
WORKDIR /app
# COPY --from=build /app/model /app/model
# COPY --from=build /app/app.py /app/
# COPY --from=build /app/requirements.txt /app/
COPY ./app.py ./requirements.txt /app/
RUN pip install -r requirements.txt
RUN mkdir model
COPY ./model/best_gradient_boosting_model.pkl /app/model

EXPOSE 5000
CMD ["python", "app.py"]


