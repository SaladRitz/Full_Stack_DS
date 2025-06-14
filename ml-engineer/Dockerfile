# start from python base image
FROM python:3.12

# change working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ./app/ /code/app/

# Expose the port Gradio/FastAPI will run on
EXPOSE 7860

# Run the app
CMD ["python", "app/main.py"]
