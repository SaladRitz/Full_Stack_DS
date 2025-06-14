## Creating Video index after Model Evaluation

# imports
import polars as pl
from sentence_transformers import SentenceTransformer

# load data
df = pl.read_parquet('data/video-transcripts.parquet')
df.head()


# embed titles and transcripts
model_name = 'all-MiniLM-L6-v2'
column_name_list = ['title', 'transcript']
model = SentenceTransformer(model_name)

for column_name in column_name_list:
    # generate embeddings
    embedding_arr = model.encode(df[column_name].to_list())

    # store embeddings in a dataframe
    schema_dict = {column_name+'_embedding-'+str(i): float for i in range(embedding_arr.shape[1])}
    df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

    # append embeddings to video index
    df = pl.concat([df, df_embedding], how='horizontal')

df.shape

df.head()

# Save index to file
df.write_parquet('data/video-index.parquet')
#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

import polars as pl
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import gradio as gr
import boto3
import json

# Load data
df_index = pl.read_parquet('data/video-index.parquet')
df_transcripts = pd.read_parquet('data/video-transcripts.parquet')

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
dist = DistanceMetric.get_metric('manhattan')

# Semantic Search
def returnSearchResults(query: str):
    query_embedding = model.encode(query).reshape(1, -1)

    title_cols = [col for col in df_index.columns if col.startswith('title_embedding')]
    transcript_cols = [col for col in df_index.columns if col.startswith('transcript_embedding')]

    title_embeddings = np.array(df_index.select(title_cols).to_numpy())
    transcript_embeddings = np.array(df_index.select(transcript_cols).to_numpy())

    dist_title = dist.pairwise(title_embeddings, query_embedding)
    dist_transcript = dist.pairwise(transcript_embeddings, query_embedding)

    dist_arr = dist_title + dist_transcript

    threshold = 40
    top_k = 5

    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    final_indices = idx_below_threshold[idx_sorted][:top_k]
    df_results = df_index.to_pandas().iloc[final_indices][['title', 'video_id']]

    # Create HTML thumbnail cards
    html_cards = ""
    choices = []
    
    for _, row in df_results.iterrows():
        video_id = row["video_id"]
        title = row["title"]
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        html_cards += f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{thumbnail_url}" alt="Thumbnail" width="120" style="margin-right: 12px; border-radius: 8px;">
            <div>
                <b>{title}</b><br>
            </div>
        </div>
        """
        choices.append((title, f"{video_id}|||{title}"))

    return html_cards, gr.update(choices=choices, visible=True)



# Show video
def show_video_detail(video_data_str):
    video_id, title = video_data_str.split("|||")
    iframe = f'<iframe width="576" height="324" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
    return video_id, title, iframe, f"# {title}"

# Summarize
def summarize_transcript(video_id, title):
    transcript = df_transcripts[df_transcripts['video_id'] == video_id]['transcript'].values[0]
    prompt = f'''You are an expert human communicator and storyteller. You’ve just finished watching a video and reading its transcript. Your task is to summarize what the video was about in a way that feels natural, human, and emotionally engaging — like how a friend would explain it to another.

    Focus on capturing:
    The main message or topic
    Emotional tone and mood (e.g. funny, inspiring, serious, eye-opening)
    Key highlights or takeaways
    The speakers attitude or intent (if noticeable)

    Avoid sounding robotic or overly formal. Use casual, empathetic language, and feel free to reflect on how it might make someone feel. '
    {title}':\n\n{transcript}'''

    # Format the request payload using the model's native structure.
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
        },
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-premier-v1:0",
        contentType="application/json",
        accept="application/json",
        body=request)
    model_response = json.loads(response["body"].read())
    try:
        return model_response["results"][0]["outputText"]
    except Exception as e:
        return f"No summary generated. Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    selected_video = gr.State()

    with gr.Tabs():
        with gr.Tab("Search"):
            search_query = gr.Textbox(label="Search Videos")
            search_btn = gr.Button("Search")
            
            thumbnails_display = gr.HTML()  # Shows thumbnails + titles
            search_radio = gr.Radio(label="Select a Video to View Video Details & Summarize", visible=False)  # For selection
            view_button = gr.Button("View Video Details")


        with gr.Tab("Details"):
            video_display = gr.HTML()
            video_title = gr.Markdown()
            summarize_btn = gr.Button("Summarize Video")
            summary_output = gr.Textbox(label="Summary", lines=10)

    # Logic connections
        search_btn.click(
        fn=returnSearchResults,
        inputs=search_query,
        outputs=[thumbnails_display, search_radio]
    )


    view_button.click(
        fn=show_video_detail,
        inputs=search_radio,
        outputs=[selected_video, video_title, video_display, video_title]
    )


    summarize_btn.click(
        fn=lambda video_id, title: summarize_transcript(video_id, title),
        inputs=[selected_video, video_title],
        outputs=summary_output
    )

demo.launch()
