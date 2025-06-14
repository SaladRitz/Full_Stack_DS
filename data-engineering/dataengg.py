import requests
import json
import polars as pl
from my_sk import my_key

from youtube_transcript_api import YouTubeTranscriptApi

## Extract Process

# define channel ID
channel_id = 'UC1oXUA7qgs0GZc_yk46K2OQ'

# define url for API
url = 'https://www.googleapis.com/youtube/v3/search'

# initialize page token
page_token = None

# intialize list to store video data
video_record_list = []



def getVideoRecords(response: requests.models.Response) -> list:
    """
        Function to extract YouTube video data from GET request response
    """

    video_record_list = []
    
    for raw_item in json.loads(response.text)['items']:
    
        # only execute for youtube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue
        
        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']
        
        video_record_list.append(video_record)

    return video_record_list


# extract video data across multiple search result pages
while page_token != 0:
    # define parameters for API call
    params = {"key": my_key, 'channelId': channel_id, 'part': ["snippet","id"], 'order': "date", 'maxResults':50, 'pageToken': page_token}
    # make get request
    response = requests.get(url, params=params)

    # append video records to list
    video_record_list += getVideoRecords(response)

    try:
        # grab next page token
        page_token = json.loads(response.text)['nextPageToken']
    except:
        # if no next page token kill while loop
        page_token = 0


# write data to file

import os

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

pl.DataFrame(video_record_list).write_parquet('data/video-ids.parquet')
pl.DataFrame(video_record_list).write_csv('data/video-ids.csv')


def extract_text(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary
    """
    
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)


# Get Transcripts

df = pl.read_parquet('data/video-ids.parquet')
print(df.head())

transcript_text_list = []

for i in range(len(df)):

    # try to extract captions
    try:
        transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
        transcript_text = extract_text(transcript)
    # if not available set as n/a
    except:
        transcript_text = "n/a"
    
    transcript_text_list.append(transcript_text)



# add transcripts to dataframe
df = df.with_columns(pl.Series(name="transcript", values=transcript_text_list))
print(df.head())

# write data to file
df.write_parquet('data/video-transcripts.parquet')
df.write_csv('data/video-transcripts.csv')



## Transform

import polars as pl
import matplotlib.pyplot as plt

# read data
df = pl.read_parquet('data/video-transcripts.parquet')

# Check for Duplicates

# shape + unique values
print("shape:", df.shape)
print("n unique rows:", df.n_unique())
for j in range(df.shape[1]):
    print("n unique elements (" + df.columns[j] + "):", df[:,j].n_unique())

print("Total number of title characters:", sum(len(df['title'][i]) for i in range(len(df))))
print("Total number of transcript characters:", sum(len(df['transcript'][i]) for i in range(len(df))))

# Check Datatypes

# change datetime to Datetime dtype
df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
print(df.head())

# Handling special strings manually by seeing output

# Before operation

print(df['title'][3])
print(df['transcript'][3])

# Operating

special_strings = ['&#39;', '&amp;', 'sha ']
special_string_replacements = ["'", "&", "Shaw "]

for i in range(len(special_strings)):
    df = df.with_columns(df['title'].str.replace(special_strings[i], special_string_replacements[i]).alias('title'))
    df = df.with_columns(df['transcript'].str.replace(special_strings[i], special_string_replacements[i]).alias('transcript'))

# After Operation

print(df['title'][3])
print(df['transcript'][3])

## Loading
# write data to file
df.write_parquet('data/video-transcripts.parquet')
df.write_csv('data/video-transcripts.csv')