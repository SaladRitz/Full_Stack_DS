# Full Stack Data Science Project
## Youtube Video Semantic Search & Summarization App

Firstly to Scrape Data I have used Youtube Transcript Api.
In the Data-engineering/dataengg.py, under the define channel ID section, put the channel ID of the youtube creator/channel you want to scrape the transcript data from.
And in the Data-engineering/my_sk.py put your youtubeAPI which can be derived from your google cloud console. (Dont share any PII on a public platform)

Now clean your data by seeing through the csv you are getting for better results

Now we come to the Data Science section where u compare different Sentence Transformer Model to see the embedding performance through similarity scores and other metrics and choose the best.
In the data-science/datasci_test.py's first section I have performed such and finally I am using all-MiniLM-L6-v2 model.

Now run locally your project using GRADIO UI interface by running the datasci_final.py file in the data-science folder. Here I have also integrated the logic of generating video-index.parquet file which will be saved in the Data folder. This will be needed to run the app. So u need to run the datasci_final.py once to generate it. Also keep in mind that I have integrated AWS bedrock model for video summarization, so set up ur AWS-CLI before running the program and also configure/edit the summarize_transcript function accoring to your bedrock model and region. Here i have used amazon.titan-text-premier-v1:0.

Now Coming to the ML-Engineer folder, here I have created a search API with FASTAPI in the app folder. Inside the utils.py you have a summarize_transcript function, configure it according to your bedrock model and region. If you have followed every step till now, it will be just fine for you to run this app. Just run the main.py in this case. GO to http://127.0.0.1:7860/ to see your results.

Search relevent videos to get search result and scroll down to select the radio button beside the option you want a video summary from and click the video details button. Since I am using gradio UI so redirecting to a page automatically isn't possible. So u need to scroll up after clicking on view video details button and go to the details tab present in the top left corner of the page. Now in the new page you have the option to click the Summarize button and it will call the bedrock model to summarize the video transcript for you.

PS: You can change the prompt in summarize_transcript function under ML-engineer/app/utils.py file and customize it as it suits you.

After this I have created a dockerfile and created a docker image for the Search API that I had made using FASTAPI. And I pushed the docker image to DockerHub. From there you can deploy it on AWS ECS or other cloud platforms and can test it. (This is a sample test url: http://<your-ec2-public-ip>:7860)




![Image](https://github.com/user-attachments/assets/761f6335-ed1b-4b9d-9a96-2976d2bdb502)




![Image](https://github.com/user-attachments/assets/190e369e-c141-419d-a611-e960d4213f27)





![Image](https://github.com/user-attachments/assets/35cbe8b5-742f-4bd3-bf2f-9876a6335df7)
