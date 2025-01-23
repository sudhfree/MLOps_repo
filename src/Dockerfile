FROM python:3.11.9 AS build
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

FROM python:3.11.9 AS deploy
WORKDIR /app
COPY --from=build /app/model /app/
COPY --from=build /app/app.py /app/
COPY --from=build /app/requirements.txt /app/
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]


