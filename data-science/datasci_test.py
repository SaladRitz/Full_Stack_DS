## Comparing Canidate Search Approaches

import polars as pl

from sentence_transformers import SentenceTransformer, util

from sklearn.metrics import DistanceMetric
import numpy as np

import matplotlib.pyplot as plt


import os

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

# load data
df = pl.read_parquet('data/video-transcripts.parquet')
df_eval = pl.read_csv('data/eval-raw.csv')
df.head()

# embed titles and transcripts

# define "parameters"
column_to_embed_list = ['title', 'transcript']
model_name_list = ["all-MiniLM-L6-v2", "multi-qa-distilbert-cos-v1", "multi-qa-mpnet-base-dot-v1"]


# generate embeddings for each combination of column and model

# initialize dict to keep track of all text embeddings
text_embedding_dict = {}

for model_name in model_name_list:

    #define embedding model
    model = SentenceTransformer(model_name) 

    for column_name in column_to_embed_list:

        # define text embedding identifier
        key_name = model_name + "_" + column_name
        print(key_name)

        # generate embeddings for text under column_name
        embedding_arr = model.encode(df[column_name].to_list())  #use %time to see compute time in jupyter notebook
        print('')

        # append embeddings to dict
        text_embedding_dict[key_name] = embedding_arr


# embed queries

query_embedding_dict = {}

for model_name in model_name_list:

    #define embedding model
    model = SentenceTransformer(model_name)
    print(model_name)

    # embed query text
    embedding_arr = model.encode(df_eval['query'].to_list())   #use %time to see compute time in jupyter notebook
    print('')

    # append embedding to dict
    query_embedding_dict[model_name] = embedding_arr



# Evaluate search Results

def returnVideoID_index(df: pl.dataframe.frame.DataFrame, df_eval: pl.dataframe.frame.DataFrame, query_n: int) -> int:
    """
        Function to return the index of a dataframe corresponding to the nth row in evaluation dataframe
    """

    return [i for i in range(len(df)) if df['video_id'][i]==df_eval['video_id'][query_n]][0]
def evalTrueRankings(dist_arr_isorted: np.ndarray, df: pl.dataframe.frame.DataFrame, df_eval: pl.dataframe.frame.DataFrame) -> np.ndarray:
    """
        Function to return "true" video ID rankings for each evaluation query
    """
    
    # intialize array to store rankings of "correct" search result
    true_rank_arr = np.empty((1, dist_arr_isorted.shape[1]))
    
    # evaluate ranking of correct result for each query
    for query_n in range(dist_arr_isorted.shape[1]):
    
        # return "true" video ID's in df
        video_id_idx = returnVideoID_index(df, df_eval, query_n)
        
        # evaluate the ranking of the "true" video ID
        true_rank = np.argwhere(dist_arr_isorted[:,query_n]==video_id_idx)[0][0]
        
        # store the "true" video ID's ranking in array
        true_rank_arr[0,query_n] = true_rank

    return true_rank_arr


# initialize distance metrics to experiment
dist_name_list = ['euclidean', 'manhattan', 'chebyshev']
sim_name_list = ['cos_sim', 'dot_score']


# evaluate all possible combinations of model, columns to embed, and distance metrics

# initialize list to store results
eval_results = []

# loop through all models
for model_name in model_name_list:

    # generate query embedding
    query_embedding = query_embedding_dict[model_name]
    
    # loop through text columns
    for column_name in column_to_embed_list:

        # generate column embedding
        embedding_arr = text_embedding_dict[model_name+'_'+column_name]

        # loop through distance metrics
        for dist_name in dist_name_list:

            # compute distance between video text and query
            dist = DistanceMetric.get_metric(dist_name)
            dist_arr = dist.pairwise(embedding_arr, query_embedding)

            # sort indexes of distance array
            dist_arr_isorted = np.argsort(dist_arr, axis=0)

            # define label for search method
            method_name = "_".join([model_name, column_name, dist_name])

            # evaluate the ranking of the ground truth
            true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

            # store results
            eval_list = [method_name] + true_rank_arr.tolist()[0]
            eval_results.append(eval_list)

        # loop through sbert similarity scores
        for sim_name in sim_name_list:
            # apply similarity score from sbert
            cmd = "dist_arr = -util." + sim_name + "(embedding_arr, query_embedding)"
            exec(cmd)
    
            # sort indexes of distance array (notice minus sign in front of cosine similarity)
            dist_arr_isorted = np.argsort(dist_arr, axis=0)
    
            # define label for search method
            method_name = "_".join([model_name, column_name, sim_name.replace("_","-")])
    
            # evaluate the ranking of the ground truth
            true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)
    
            # store results
            eval_list = [method_name] + true_rank_arr.tolist()[0]
            eval_results.append(eval_list)


# compute rankings for title + transcripts embedding
for model_name in model_name_list:
    
    # generate embeddings
    embedding_arr1 = text_embedding_dict[model_name+'_title']
    embedding_arr2 = text_embedding_dict[model_name+'_transcript']
    query_embedding = query_embedding_dict[model_name]

    for dist_name in dist_name_list:

        # compute distance between video text and query
        dist = DistanceMetric.get_metric(dist_name)
        dist_arr = dist.pairwise(embedding_arr1, query_embedding) + dist.pairwise(embedding_arr2, query_embedding)

        # sort indexes of distance array
        dist_arr_isorted = np.argsort(dist_arr, axis=0)

         # define label for search method
        method_name = "_".join([model_name, "title-transcript", dist_name])

        # evaluate the ranking of the ground truth
        true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

        # store results
        eval_list = [method_name] + true_rank_arr.tolist()[0]
        eval_results.append(eval_list)

    # loop through sbert similarity scores
    for sim_name in sim_name_list:
        # apply similarity score from sbert
        cmd = "dist_arr = -util." + sim_name + "(embedding_arr1, query_embedding) - util."+ sim_name + "(embedding_arr2, query_embedding)"
        exec(cmd)

        # sort indexes of distance array (notice minus sign in front of cosine similarity)
        dist_arr_isorted = np.argsort(dist_arr, axis=0)

        # define label for search method
        method_name = "_".join([model_name, "title-transcript", sim_name.replace("_","-")])

        # evaluate the ranking of the ground truth
        true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

        # store results
        eval_list = [method_name] + true_rank_arr.tolist()[0]
        eval_results.append(eval_list)

len(eval_results)

# define schema for results dataframe
schema_dict = {'method_name':str}
for i in range(len(eval_results[0])-1):
    schema_dict['rank_query-'+str(i)] = float

# store results in dataframe
df_results = pl.DataFrame(eval_results, schema=schema_dict)
df_results.head()


# compute mean rankings of ground truth search result
df_results = df_results.with_columns(new_col=pl.mean_horizontal(df_results.columns[1:])).rename({"new_col": "rank_query-mean"})


# compute number of ground truth results which appear in top 3
for i in [1,3]:
    df_results = df_results.with_columns(new_col=pl.sum_horizontal(df_results[:,1:-1]<i)).rename({"new_col": "num_in_top-"+str(i)})


# Look at top results
df_summary = df_results[['method_name', "rank_query-mean", "num_in_top-1", "num_in_top-3"]]

print(df_summary.sort('rank_query-mean').head())

df_summary.sort("num_in_top-1", descending=True).head()[0,0]

print(df_summary.sort("num_in_top-3", descending=True).head())

df_summary.sort("num_in_top-3", descending=True).head()[0,0]

for i in range(4):
    print(df_summary.sort("num_in_top-3", descending=True)['method_name'][i])



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


## Semantic Search Function

import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
import gradio as gr

# load data, model, and metric
df = pl.scan_parquet('data/video-index.parquet')

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)



# search function
def returnSearchResults(query: str, index: pl.lazyframe.frame.LazyFrame) -> np.ndarray:
    """
        Function to return indexes of top search results
    """
    
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)

    # Get column names without triggering schema resolution warning
    #column_names = index.collect_schema().names()
    column_names = list(index.schema.keys())

    
    # compute distances between query and titles/transcripts
    dist_arr = (
        dist.pairwise(index.select(column_names[4:388]).collect(), query_embedding) +
        dist.pairwise(index.select(column_names[388:]).collect(), query_embedding)
    )

    # search paramaters
    threshold = 40 # eye balled threshold for manhatten distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten()<threshold).flatten()
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # return indexes of search results
    return idx_below_threshold[idx_sorted][:top_k]

query = "LLM"
idx_result = returnSearchResults(query, df)

print(df.select(['video_id', 'title']).collect()[idx_result])

df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)

# Interface

def pseudoSearchAPI(query: str):
    
    # return top 5 search results
    idx_result = returnSearchResults(query, df)
    response = df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)

    return response
def formatResultText(title: str, video_id: str):
    
    text = markdown_text = f"""<br> <br>
# {title}<br>

🔗 [Video Link](https://youtu.be/{video_id})"""

    return text
def formatVideoEmbed(video_id: str):

    # other options
    # embed = '<iframe width="640" height="360" src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" </iframe>'
    # embed = '<a href="https://youtu.be/'+ video_id +'"> <img src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" style="width:576;height:324;"></a>'
    # embed = '<a href="www.youtube.com/watch?v='+ video_id +'"> <img src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" style="width:576;height:324;"></a>'
    
    return '<iframe width="576" height="324" src="https://www.youtube.com/embed/'+ video_id +'"></iframe>'
def searchResults(query):
    # pseudo API call
    response = pseudoSearchAPI(query)

    # format search results

    # initialize list of outputs
    output_list = []

    # compute number of null search results (out of 5)
    num_empty_results = 5-len(response['title'])

    # display search results
    for i in range(len(response['title'])):
        video_id = response['video_id'][i]
        title = response['title'][i]

        embed = gr.HTML(value = formatVideoEmbed(video_id), visible=True)
        text = gr.Markdown(value = formatResultText(title, video_id), visible=True)

        output_list.append(embed)
        output_list.append(text)

    # make null search result slots invisible
    for i in range(num_empty_results):
        
        # if no search results display "No results." text
        if num_empty_results==5 and i==0:
            embed = gr.HTML(visible=False)
            text = gr.Markdown(value = "No results. Try rephrasing your query.", visible=True)

            output_list.append(embed)
            output_list.append(text)
            continue

        embed = gr.HTML(visible=False)
        text = gr.Markdown(visible=False)

        output_list.append(embed)
        output_list.append(text)
        
    return output_list
# demo
output_list = []

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Search")

    with gr.Row():
        inp = gr.Textbox(placeholder="What are you looking for?", label="Query", scale=3)
        btn = gr.Button("Search")
        btn.click(fn=searchResults, inputs=inp, outputs=output_list)
    
    for i in range(5):
        with gr.Row():
            output_list.append(gr.HTML())
            output_list.append(gr.Markdown())
             
    inp.submit(fn=searchResults, inputs=inp, outputs=output_list)

demo.launch()
