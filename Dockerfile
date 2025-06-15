# start from python base image
FROM python:3.12

# change working directory
WORKDIR /code

# add requirements file to image
COPY ml-engineer/requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ml-engineer/app/ /code/app/

# Expose the port Gradio/FastAPI will run on
#EXPOSE 7860
EXPOSE 8000

# Run the backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
