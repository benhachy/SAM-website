# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

EXPOSE 8000

## api-transport-https installation
RUN apt-get install apt-transport-https ca-certificates gnupg

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

RUN python manage.py makemigrations 
RUN python manage.py migrate
RUN python manage.py collectstatic --no-input

ENTRYPOINT ["gunicorn", "myteam.wsgi:application", "--bind=0.0.0.0:8000", "--workers=4", "--timeout=300", "--log-level=debug"]