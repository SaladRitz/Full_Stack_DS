# Full Stack Data Science Project
## Youtube Video Semantic Search & Summarization App

Firstly to Scrape Data I have used Youtube Transcript Api.
In the Data-engineering/dataengg.py, under the define channel ID section, put the channel ID of the youtube creator/channel you want to scrape the transcript data from.
And in the Data-engineering/my_sk.py put your youtubeAPI which can be derived from your google cloud console. (Dont share any PII on a public platform)

Now clean your data by seeing through the csv you are getting for better results

Now we come to the Data Science section where u compare different Sentence Transformer Model to see the embedding performance through similarity scores and other metrics and choose the best.
In the data-science/datasci_test.py's first section I have performed such and finally I am using all-MiniLM-L6-v2 model.

Now run locally your project using GRADIO UI interface by running the datasci_final.py file in the data-science folder. Here I have also integrated the logic of generating video-index.parquet file which will be saved in the Data folder. This will be needed to run the app. So u need to run the datasci_final.py once to generate it. Also keep in mind that I have integrated AWS bedrock model for video summarization, so set up ur AWS-CLI before running the program and also configure/edit the summarize_transcript function accoring to your bedrock model and region. Here i have used amazon.titan-text-premier-v1:0.

Now coming to the ml-engineer folder, which is the core of my full-stack ML application where all the backend and frontend logic lives. Inside the app/ directory, I’ve built a FastAPI-based backend that handles key features like semantic search over YouTube video transcripts and generating human-like summaries using AWS Bedrock. To make the backend portable and easy to deploy, I created a Docker image of the FastAPI server, then pushed that image to Docker Hub. After that, I deployed it on AWS ECS (Elastic Container Service), which allows the backend to run as a containerized service in the cloud. This means the backend is live and can be accessed via a public URL, even when my local machine is turned off.

On the other hand, the frontend/ directory contains a gradio_ui.py file, which sets up a user-friendly interface using Gradio. This Gradio frontend is what I use locally for testing and interacting with the backend. It lets me enter a query, perform semantic search over indexed transcripts, select a result, and trigger the summarization feature — all while the actual heavy lifting happens on the backend running in AWS. This setup allows for a clean separation between backend services and the local frontend, giving me the flexibility to iterate quickly without constantly redeploying the UI. It also mimics a real-world production setup where the backend is hosted remotely and the frontend can be run anywhere.


To use the app as i have intended, use the dockerfile that i have provided to build a docker image:
In your project repo open a terminal (I have used VS code) and type in the command:
docker build -t {your_dockerhub_username}/video-search-api .

Push to DockerHub by the commands:
docker login
docker tag {your_dockerhub_username}/video-search-app {your_dockerhub_username}/video-search-app:latest
docker push {your_dockerhub_username}/video-search-app:latest


To deploy the app on an EC2 instance, I launched an Ubuntu server (t2.medium or higher) and allowed ports 22 (SSH) and 8000 (FAST API backend), in the security group. After SSH-ing into the instance with an ec2-key.pem file, I installed Docker, started and enabled the Docker service, then pulled my image from Docker Hub using docker pull. Finally, I ran the container with port 8000 exposed, on which the FastAPI server runs on, and made sure the security group allows traffic on that port. This lets the backend be publicly accessible via the ECS public IP.

Now if u go to http://<EC2_PUBLIC_IP>:8000, you will see {"message":"FastAPI is working!"} if everything works fine.

(IMPORTANT) To run the app locally open a terminal in your project repo and run:
uvicorn ml-engineer.app.main:app --reload --port 8000
This starts the FAST API backend.
Now open another terminal and run:
python ml-engineer\frontend\gradio_ui.py

OR simply run the gradio_ui.py in your vs code. And go to http://127.0.0.1:7860 to see Results.




## Semantic Search Results




![Image](https://github.com/user-attachments/assets/761f6335-ed1b-4b9d-9a96-2976d2bdb502)




![Image](https://github.com/user-attachments/assets/190e369e-c141-419d-a611-e960d4213f27)





## Summarization Result





![Image](https://github.com/user-attachments/assets/35cbe8b5-742f-4bd3-bf2f-9876a6335df7)





