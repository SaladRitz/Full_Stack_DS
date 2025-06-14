from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(video_id_or_url):
    # a youtube video id is 11 characters long
    # if the video id is longer than that, then it's a url
    if len(video_id_or_url) > 11:
        # it's a url
        # the video id is the last 11 characters
        return video_id_or_url[-11:]
    else:
        # it's a video id
        return video_id_or_url
def get_transcript(video_url_or_id):
    try:
        video_id = extract_video_id(video_url_or_id)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None


video_url = 'https://www.youtube.com/watch?v=uLrReyH5cu0'
transcript = get_transcript(video_url)

print(transcript)