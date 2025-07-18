import polars as pl
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import boto3
import json

# Load data
df_index = pl.read_parquet('data/video-index.parquet')
df_transcripts = pd.read_parquet('data/video-transcripts.parquet')

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
dist = DistanceMetric.get_metric('manhattan')


def search_videos(query: str, top_k: int = 5, threshold: float = 40):
    query_embedding = model.encode(query).reshape(1, -1)
    title_cols = [col for col in df_index.columns if col.startswith('title_embedding')]
    transcript_cols = [col for col in df_index.columns if col.startswith('transcript_embedding')]
    
    title_embeddings = np.array(df_index.select(title_cols).to_numpy())
    transcript_embeddings = np.array(df_index.select(transcript_cols).to_numpy())
    
    dist_title = dist.pairwise(title_embeddings, query_embedding)
    dist_transcript = dist.pairwise(transcript_embeddings, query_embedding)
    dist_arr = dist_title + dist_transcript
    
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()
    
    final_indices = idx_below_threshold[idx_sorted][:top_k]
    df_results = df_index.to_pandas().iloc[final_indices][['title', 'video_id']]
    return df_results.to_dict(orient='records')


def summarize_transcript(video_id: str, title: str) -> str:
    transcript = df_transcripts[df_transcripts['video_id'] == video_id]['transcript'].values[0]
    prompt = f'''You are an expert human communicator and storyteller. You’ve just finished watching a video and reading its transcript. Your task is to summarize what the video was about in a way that feels natural, human, and emotionally engaging — like how a friend would explain it to another.

Focus on capturing:
- The main message or topic
- Emotional tone and mood
- Key highlights or takeaways
- The speaker's attitude or intent

Avoid sounding robotic or overly formal. Use casual, empathetic language.
'{title}':\n\n{transcript}
'''
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
        },
    }
    request = json.dumps(native_request)

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-premier-v1:0",
        contentType="application/json",
        accept="application/json",
        body=request,
    )
    model_response = json.loads(response["body"].read())

    try:
        return model_response["results"][0]["outputText"]
    except Exception as e:
        return f"No summary generated. Error: {str(e)}"
