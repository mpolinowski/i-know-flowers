# base image to use
FROM python:3.11-slim-bookworm
# dir name inside the container used for your app
WORKDIR /opt/python_app
# copy all files into the work dir
COPY . .
# install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# execute the app when container starts
EXPOSE 5000/tcp
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0"]
