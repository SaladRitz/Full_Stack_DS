import gradio as gr
from utils import returnSearchResults, show_video_detail, summarize_transcript

def build_gradio_ui():
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

    return demo
