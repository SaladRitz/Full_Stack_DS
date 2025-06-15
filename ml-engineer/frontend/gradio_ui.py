import gradio as gr
import requests

API_URL = "http://localhost:8000"

def gradio_search(query):
    response = requests.post(f"{API_URL}/search", json={"query": query})
    results = response.json()["results"]

    html_cards = ""
    choices = []

    for result in results:
        title = result["title"]
        video_id = result["video_id"]
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        html_cards += f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{thumbnail_url}" alt="Thumbnail" width="120" style="margin-right: 12px; border-radius: 8px;">
            <div><b>{title}</b></div>
        </div>
        """
        choices.append((title, f"{video_id}|||{title}"))

    return html_cards, gr.update(choices=choices, visible=True)

def show_video_detail(video_data_str):
    video_id, title = video_data_str.split("|||")
    iframe = f'<iframe width="576" height="324" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
    return video_id, title, iframe, f"# {title}"

def summarize_from_api(video_id, title):
    response = requests.post(f"{API_URL}/summarize", json={"video_id": video_id, "title": title})
    return response.json()["summary"]

with gr.Blocks() as demo:
    selected_video = gr.State()

    with gr.Tabs():
        with gr.Tab("Search"):
            search_query = gr.Textbox(label="Search Videos")
            search_btn = gr.Button("Search")
            thumbnails_display = gr.HTML()
            search_radio = gr.Radio(label="Select a Video", visible=False)
            view_button = gr.Button("View Video Details")

        with gr.Tab("Details"):
            video_display = gr.HTML()
            video_title = gr.Markdown()
            summarize_btn = gr.Button("Summarize Video")
            summary_output = gr.Textbox(label="Summary", lines=10)

    search_btn.click(
        fn=gradio_search,
        inputs=search_query,
        outputs=[thumbnails_display, search_radio]
    )

    view_button.click(
        fn=show_video_detail,
        inputs=search_radio,
        outputs=[selected_video, video_title, video_display, video_title]
    )

    summarize_btn.click(
        fn=summarize_from_api,
        inputs=[selected_video, video_title],
        outputs=summary_output
    )

demo.launch()
